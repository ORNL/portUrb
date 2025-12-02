## Home for all external git submodules

* `eigen`: For convenient linear algebra operations. These should generally be performed on the host because generally only CUDA is supported for GPU computations.
* `kokkos`: This is the main workhorse of the portability among CPUs and GPUs. Generally, kokkos is used through the Yet Another Kernel Launcher (YAKL) library, but you're also of course free to use Kokkos directly if you wish.
* `ponni`: Portable Online Neural Network Inferencing (PONNI). This is a nascent simple library for creating basic neural networks in templated C++ and reading weights from Keras, Tensorflow, and pyTorch HDF5 files. Relatively few types of layers are supported at present, though.
* `WGS84toCartesian`: This is to provide convenient translation between Cartesian East-North-Up coordinates and WGS84 global latitude longitude coordinates.
* `YAKL`: YAKL is A Kokkos Layer (YAKL). This is a relatively thin layer on top of Kokkos to support Fortran-style array indexing when desires, an automatic and transparent pool allocator, and some other convenience functions as well. Most of the code here uses YAKL, which itself is built on top of kokkos, rather than kokkos directly.
* `yaml-cpp`: YAML Ain't Markup Language (YAML). YAML is my favorite language for input files, and I encourage you to use it as well for input file in your experiments. You can do a lot of convenient stuff in YAML like input artibrarily typed arrays of arrays (of arrays).
