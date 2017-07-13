package hyperloglog

import (
	"bytes"
	"encoding/binary"
	"encoding/gob"
	"errors"
	"fmt"

	bits "github.com/dgryski/go-bits"
)

const pPrime = 25
const mPrime = 1 << (pPrime - 1)

var threshold = []uint{
	10, 20, 40, 80, 220, 400, 900, 1800, 3100,
	6500, 11500, 20000, 50000, 120000, 350000,
}

type HyperLogLogPlus struct {
	reg       []uint8
	p         uint8
	m         uint32
	sparse    bool
	sparseSet compactSet
}

// Encode a hash to be used in the sparse representation.
func (h *HyperLogLogPlus) encodeHash(x uint64) uint32 {
	idx := uint32(eb64(x, 64, 64-pPrime))

	if eb64(x, 64-h.p, 64-pPrime) == 0 {
		zeros := bits.Clz((eb64(x, 64-pPrime, 0)<<pPrime)|(1<<pPrime-1)) + 1
		return idx<<7 | uint32(zeros<<1) | 1
	}
	return idx << 1
}

// Get the index of precision p from the sparse representation.
func (h *HyperLogLogPlus) getIndex(k uint32) uint32 {
	if k&1 == 1 {
		return eb32(k, 32, 32-h.p)
	}
	return eb32(k, pPrime+1, pPrime-h.p+1)
}

// Decode a hash from the sparse representation.
func (h *HyperLogLogPlus) decodeHash(k uint32) (uint32, uint8) {
	var r uint8
	if k&1 == 1 {
		r = uint8(eb32(k, 7, 1)) + pPrime - h.p
	} else {
		r = clz32(k<<(32-pPrime+h.p-1)) + 1
	}
	return h.getIndex(k), r
}

// NewPlus returns a new initialized HyperLogLogPlus that uses the HyperLogLog++
// algorithm.
func NewPlus(precision uint8) (*HyperLogLogPlus, error) {
	h, err := createPlus(precision)
	if err != nil {
		return nil, err
	}
	h.Clear()
	return h, nil
}

// createPlus returns a new, empty HyperLogLogPlus w/o inititalizing internal
// buffers, for use as a deserialization target.
func createPlus(precision uint8) (*HyperLogLogPlus, error) {
	if precision > 18 || precision < 4 {
		return nil, errors.New("precision must be between 4 and 18")
	}

	h := &HyperLogLogPlus{}
	h.p = precision
	h.m = 1 << precision
	return h, nil
}

// Clear sets HyperLogLogPlus h back to its initial state.
func (h *HyperLogLogPlus) Clear() {
	h.sparse = true
	h.sparseSet = make(compactSet, 0, h.m/4)
	h.reg = nil
}

func (h *HyperLogLogPlus) maxTmpSet() int {
	return int(h.m) / 100
}

// Converts HyperLogLogPlus h to the normal representation from the sparse
// representation.
func (h *HyperLogLogPlus) toNormal() {
	h.reg = make([]uint8, h.m)
	for _, k := range h.sparseSet {
		h.addHash(k)
	}

	h.sparse = false
	h.sparseSet = nil
}

func (h *HyperLogLogPlus) addHash(k uint32) {
	i, r := h.decodeHash(k)
	if h.reg[i] < r {
		h.reg[i] = r
	}
}

// Add adds a new item to HyperLogLogPlus h.
func (h *HyperLogLogPlus) Add(item Hash64) {
	x := item.Sum64()
	if h.sparse {
		h.sparseSet.Add(h.encodeHash(x))
		h.maybeToNormal()
	} else {
		i := eb64(x, 64, 64-h.p) // {x63,...,x64-p}
		w := x<<h.p | 1<<(h.p-1) // {x63-p,...,x0}

		zeroBits := uint8(bits.Clz(w)) + 1
		if zeroBits > h.reg[i] {
			h.reg[i] = zeroBits
		}
	}
}

// Merge takes another HyperLogLogPlus and combines it with HyperLogLogPlus h.
func (h *HyperLogLogPlus) Merge(other *HyperLogLogPlus) error {
	if h.p != other.p {
		return errors.New("precisions must be equal")
	}

	if h.sparse && other.sparse {
		origSparse := h.sparseSet[:]
		for _, k := range other.sparseSet {
			if !h.sparse {
				h.addHash(k)
				continue
			}

			// Optimization: other.sparseSet is already de-duped, so only check
			// for dupes against our original, local sparseSet
			if !origSparse.Has(k) {
				h.sparseSet = append(h.sparseSet, k)
			}
			h.maybeToNormal()
		}
		return nil
	}

	if h.sparse {
		h.toNormal()
	}

	if other.sparse {
		for _, k := range other.sparseSet {
			h.addHash(k)
		}
	} else {
		for i, v := range other.reg {
			if v > h.reg[i] {
				h.reg[i] = v
			}
		}
	}
	return nil
}

// Converts to normal if the sparse list is too large.
func (h *HyperLogLogPlus) maybeToNormal() {
	if uint32(len(h.sparseSet)) >= h.m/4 {
		h.toNormal()
	}
}

// Estimates the bias using empirically determined values.
func (h *HyperLogLogPlus) estimateBias(est float64) float64 {
	estTable, biasTable := rawEstimateData[h.p-4], biasData[h.p-4]

	if estTable[0] > est {
		return biasTable[0]
	}

	lastEstimate := estTable[len(estTable)-1]
	if lastEstimate < est {
		return biasTable[len(biasTable)-1]
	}

	var i int
	for i = 0; i < len(estTable) && estTable[i] < est; i++ {
	}

	e1, b1 := estTable[i-1], biasTable[i-1]
	e2, b2 := estTable[i], biasTable[i]

	c := (est - e1) / (e2 - e1)
	return b1*(1-c) + b2*c
}

// Count returns the cardinality estimate.
func (h *HyperLogLogPlus) Count() uint64 {
	if h.sparse {
		return uint64(linearCounting(mPrime, mPrime-uint32(len(h.sparseSet))))
	}

	est := calculateEstimate(h.reg)
	if est <= float64(h.m)*5.0 {
		est -= h.estimateBias(est)
	}

	if v := countZeros(h.reg); v != 0 {
		lc := linearCounting(h.m, v)
		if lc <= float64(threshold[h.p-4]) {
			return uint64(lc)
		}
	}
	return uint64(est)
}

// Encode HyperLogLogPlus into a gob
func (h *HyperLogLogPlus) GobEncode() ([]byte, error) {
	buf := bytes.Buffer{}
	enc := gob.NewEncoder(&buf)
	if err := enc.Encode(h.reg); err != nil {
		return nil, err
	}
	if err := enc.Encode(h.m); err != nil {
		return nil, err
	}
	if err := enc.Encode(h.p); err != nil {
		return nil, err
	}
	if err := enc.Encode(h.sparse); err != nil {
		return nil, err
	}
	if h.sparse {
		if err := enc.Encode(h.sparseSet); err != nil {
			return nil, err
		}
	}
	return buf.Bytes(), nil
}

// Decode gob into a HyperLogLogPlus structure
func (h *HyperLogLogPlus) GobDecode(b []byte) error {
	dec := gob.NewDecoder(bytes.NewBuffer(b))
	if err := dec.Decode(&h.reg); err != nil {
		return err
	}
	if err := dec.Decode(&h.m); err != nil {
		return err
	}
	if err := dec.Decode(&h.p); err != nil {
		return err
	}
	if err := dec.Decode(&h.sparse); err != nil {
		return err
	}
	if h.sparse {
		if err := dec.Decode(&h.sparseSet); err != nil {
			return err
		}
	}
	return nil
}

const binaryVersion = 1

// Encode to binary much faster (and less safely) than Gob can manage.
// Implements the encoding.BinaryMarshaler interface.
func (h *HyperLogLogPlus) MarshalBinary() ([]byte, error) {
	size := 3
	if h.sparse {
		size += 4 * len(h.sparseSet)
	} else {
		size += len(h.reg)
	}

	data := make([]byte, size)
	data[0] = binaryVersion
	data[1] = h.p

	if h.sparse {
		data[2] = 1
		for i, val := range h.sparseSet {
			offset := 3 + (i * 4)
			binary.BigEndian.PutUint32(data[offset:offset+4], val)
		}
		return data, nil
	}

	copy(data[3:], h.reg)

	return data, nil
}

// Decode binary created by MarshalBinary, above.
// Can safely be called on an empty HyperLogLogPlus struct.
// Implements the encoding.BinaryUnmarshaler interface.
func (h *HyperLogLogPlus) UnmarshalBinary(data []byte) error {
	if data[0] != binaryVersion {
		return fmt.Errorf("cannot unmarshal unknown binary version %d", data[0])
	}

	if data[1] > 18 || data[1] < 4 {
		return fmt.Errorf("cannot unmarshal invalid p %d", data[1])
	}
	h.p = data[1]
	h.m = 1 << h.p
	h.sparse = data[2] == 1

	if h.sparse {
		h.reg = nil

		if len(data) > int(h.m)+3 {
			return fmt.Errorf("expected buffer of max size %d, got %d", h.m+3, len(data))
		}

		h.sparseSet = make(compactSet, 0, h.m/4)
		for i := 3; i+4 <= len(data); i += 4 {
			h.sparseSet = append(h.sparseSet, binary.BigEndian.Uint32(data[i:i+4]))
		}

		return nil
	}

	h.sparseSet = nil
	if len(data) < int(h.m)+3 {
		return fmt.Errorf("expected buffer of size %d, got %d", h.m+3, len(data))
	}

	h.reg = make([]uint8, h.m)
	copy(h.reg, data[3:])
	return nil
}
