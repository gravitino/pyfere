#include <boost/python.hpp>
#include <cstdint>
#include <atomic>
#include <vector>
#include <random>

namespace p  = boost::python;

template <uint64_t seed=42>
struct fmix64_t {

    typedef uint64_t data_t;
    typedef uint64_t rtrn_t;
    static constexpr uint64_t width = sizeof(data_t) << 3;

    rtrn_t operator()(const data_t x, const uint64_t k) const {

        #define BIG_CONSTANT(x) (x##LLU)

        rtrn_t y = x+k+seed;

        y ^= y >> 33;
        y *= BIG_CONSTANT(0xff51afd7ed558ccd);
        y ^= y >> 33;
        y *= BIG_CONSTANT(0xc4ceb9fe1a85ec53);
        y ^= y >> 33;

        return y;
    }

    rtrn_t mod_width(const rtrn_t x) const {
        return x & 63;
    }
};
#include<iostream>
template <uint64_t seed=42, uint64_t max_k=128>
struct tmix64_t {

    typedef uint64_t data_t;
    typedef uint64_t rtrn_t;
    static constexpr uint64_t width = sizeof(data_t) << 3;

    std::vector<uint64_t> table_range;

    tmix64_t () {

        std::mt19937 engine(seed);
        std::uniform_int_distribution<uint64_t> range;

        for (uint64_t k = 0; k < max_k; k++)
            for (uint64_t slot = 0; slot < 8; slot++)
                for (uint64_t entry = 0; entry < 256; entry++)
                    table_range.push_back(range(engine));
    }

    rtrn_t operator()(const data_t x, const uint64_t k) const {

        rtrn_t result = 0;
        for (uint64_t slot = 0; slot < 8; slot++) {
            const uint64_t entry = (x >> (8*slot)) & 255;
            result ^= table_range[k*256*8+slot*256+entry];
        }

        return result;
    }

    rtrn_t mod_width(const rtrn_t x) const {
        return x & 63;
    }
};

template <class data_t>
struct Bitarray {

    static constexpr uint64_t width = sizeof(data_t) << 3;

    std::atomic<data_t> * data;

    const uint64_t capacity;
    const uint64_t num_slots;
    const bool in_parallel;

    Bitarray(const uint64_t capacity_,
             const bool in_parallel_=false) :
        capacity    (capacity_),
        num_slots   ((capacity_+width-1)/width),
        in_parallel (in_parallel_) {

        data = new std::atomic<data_t>[num_slots];

        # pragma omp parallel for if (in_parallel)
        for (uint64_t slot = 0; slot < num_slots; slot++)
            data[slot].store(0);
    }

    ~Bitarray () {
        delete [] data;
    }

    template <bool throw_exception=false>
    void set(const uint64_t pos) {

        if (pos >= capacity) {
            if (throw_exception)
                throw std::out_of_range("Index out of range");
            else
                return;
        }

        const uint64_t slot = pos/width;
        const uint64_t mask = 1UL << (pos-slot*width);

        const data_t expected = data[slot].load();

        if (!(expected & mask))
            data[slot].fetch_or(mask);

    }

    template <bool throw_exception=false>
    bool get(const uint64_t pos) {

        if (pos >= capacity) {
            if (throw_exception)
                throw std::out_of_range("Index out of range");
            else
                return false;
        }

        const uint64_t slot = pos/width;
        const uint64_t mask = 1UL << (pos-slot*width);

        return data[slot].load() & mask;
    }

    uint64_t len () {
        return capacity;
    }
};

template <class hash_t>
struct BloomFilter{

    typedef typename hash_t::data_t data_t;
    typedef Bitarray<data_t> array_t;
    static constexpr uint64_t width = hash_t::width;

    const uint64_t capacity;
    const uint64_t num_slots;
    const uint64_t num_hashes;
    const hash_t hash;
    const bool in_parallel;

    std::atomic<data_t> * data;

    BloomFilter(
        const uint64_t capacity_,
        const uint64_t num_hashes_,
        const bool in_parallel_=false) :
            capacity    (capacity_),
            num_slots   ((capacity_+width-1)/width),
            num_hashes  (num_hashes_),
            hash        (hash_t()),
            in_parallel (in_parallel_) {

        data = new std::atomic<data_t>[num_slots];

        # pragma omp parallel for if (in_parallel)
        for (uint64_t slot = 0; slot < num_slots; slot++)
            data[slot].store(0);
    }

    ~BloomFilter () {
        delete [] data;
    }

    void insert(p::list& list) {

        const uint64_t num_pos = p::len(list);

        data_t * hashes = new data_t[num_pos];
        for (uint64_t pos = 0; pos < num_pos; pos++)
            hashes[pos] = PyObject_Hash(p::api::object(list[pos]).ptr());

        # pragma omp parallel for if (in_parallel)
        for (uint64_t pos = 0; pos < num_pos; pos++) {

            const data_t c_hash = hashes[pos];

            for (uint64_t hash_id = 0; hash_id < num_hashes; hash_id++) {
                const uint64_t loc  = hash(c_hash, hash_id) % capacity;
                const uint64_t slot = loc/width;
                const uint64_t bit  = loc-slot*width;
                const data_t   mask = 1UL << bit;
                const data_t expected = data[slot].load();

                if ((expected & mask) !=  mask)
                    data[slot].fetch_or(mask);
            }
        }

        delete [] hashes;
    }

    boost::shared_ptr<array_t> query(p::list& list) {

        const uint64_t num_pos = p::len(list);
        boost::shared_ptr<array_t> result(new array_t(num_pos, in_parallel));

        data_t * hashes = new data_t[num_pos];
        for (uint64_t pos = 0; pos < num_pos; pos++)
            hashes[pos] = PyObject_Hash(p::api::object(list[pos]).ptr());

        # pragma omp parallel for if (in_parallel)
        for (uint64_t pos = 0; pos < num_pos; pos++) {

            const data_t c_hash = hashes[pos];

            bool go = true;
            for (uint64_t hash_id = 0; hash_id < num_hashes && go; hash_id++) {
                const uint64_t loc  = hash(c_hash, hash_id) % capacity;
                const uint64_t slot = loc/width;
                const uint64_t bit  = loc-slot*width;
                const data_t   mask = 1UL << bit;
                const data_t expected = data[slot].load();

                if ((expected & mask) !=  mask)
                    go = false;
            }

            if (go)
                result-> template set<false>(pos);
        }

        delete [] hashes;

        return result;
    }
};

template <class hash_t>
struct PartitionedBloomFilter{

    typedef typename hash_t::data_t data_t;
    typedef Bitarray<data_t> array_t;
    static constexpr uint64_t width = hash_t::width;

    const uint64_t capacity;
    const uint64_t num_slots;
    const uint64_t num_hashes;
    const hash_t hash;
    const bool in_parallel;

    std::atomic<data_t> * data;

    PartitionedBloomFilter(
        const uint64_t capacity_,
        const uint64_t num_hashes_,
        const bool in_parallel_=false) :
            capacity    (capacity_),
            num_slots   ((capacity_+width-1)/width),
            num_hashes  (num_hashes_),
            hash        (hash_t()),
            in_parallel (in_parallel_) {

        data = new std::atomic<data_t>[num_slots];

        # pragma omp parallel for if (in_parallel)
        for (uint64_t slot = 0; slot < num_slots; slot++)
            data[slot].store(0);
    }

    ~PartitionedBloomFilter () {
        delete [] data;
    }

    void insert(p::list& list) {

        const uint64_t num_pos = p::len(list);

        data_t * hashes = new data_t[num_pos];
        for (uint64_t pos = 0; pos < num_pos; pos++)
            hashes[pos] = PyObject_Hash(p::api::object(list[pos]).ptr());

        # pragma omp parallel for if (in_parallel)
        for (uint64_t pos = 0; pos < num_pos; pos++) {

            const data_t c_hash = hashes[pos];

            data_t mask = 0;
            for (uint64_t hash_id = 0; hash_id < num_hashes; hash_id++)
                mask |= 1UL << hash.mod_width(hash(c_hash, hash_id));

            const uint64_t slot = hash(c_hash, num_hashes) % num_slots;
            const data_t expected = data[slot].load();

            if ((expected & mask) !=  mask)
                data[slot].fetch_or(mask);
        }

        delete [] hashes;
    }

    boost::shared_ptr<array_t> query(p::list& list) {

        const uint64_t num_pos = p::len(list);
        boost::shared_ptr<array_t> result(new array_t(num_pos, in_parallel));

        data_t * hashes = new data_t[num_pos];
        for (uint64_t pos = 0; pos < num_pos; pos++)
            hashes[pos] = PyObject_Hash(p::api::object(list[pos]).ptr());

        # pragma omp parallel for if (in_parallel)
        for (uint64_t pos = 0; pos < num_pos; pos++) {

            const data_t c_hash = hashes[pos];

            data_t mask = 0;
            for (uint64_t hash_id = 0; hash_id < num_hashes; hash_id++)
                mask |= 1UL << hash.mod_width(hash(c_hash, hash_id));

            const uint64_t slot = hash(c_hash, num_hashes) % num_slots;
            const data_t expected = data[slot].load();

            if ((expected & mask) ==  mask)
                result-> template set<false>(pos);

        }

        delete [] hashes;

        return result;
    }
};

BOOST_PYTHON_MODULE(pyfere) {

    typedef BloomFilter<fmix64_t<>> bfilter64_t;
    typedef boost::shared_ptr<bfilter64_t> bfilter64_ptr_t;

    p::class_<bfilter64_t, bfilter64_ptr_t>("BloomFilter",
                          p::init<uint64_t, uint64_t, p::optional<bool> >())
        .def("insert", &bfilter64_t::insert)
        .def("query",  &bfilter64_t::query );


    typedef PartitionedBloomFilter<fmix64_t<>> pfilter64_t;
    typedef boost::shared_ptr<pfilter64_t> pfilter64_ptr_t;

    p::class_<pfilter64_t, pfilter64_ptr_t>("PartitionedBloomFilter",
                          p::init<uint64_t, uint64_t, p::optional<bool> >())
        .def("insert", &pfilter64_t::insert)
        .def("query",  &pfilter64_t::query );

    typedef Bitarray<typename pfilter64_t::data_t> bitarray64_t;
    typedef boost::shared_ptr<bitarray64_t> bitarray64_ptr_t;

    p::class_<bitarray64_t, bitarray64_ptr_t>("Bitarray",
                            p::init<uint64_t, p::optional<bool> >())
        .def("__getitem__", &bitarray64_t::get<true>)
        .def("__len__",     &bitarray64_t::len);
}
