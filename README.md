# TorchSLAM

Author: Zegang Cheng

**REMAINDER** The project is in alpha stage, and it is still under heavily development, so the performance is not guaranteed.

This is a Differentiable Distributed Visual-SLAM (Simultaneous localization and mapping) System built on Pytorch. It is NOT aimed for embedded system, on the contrary, the project is designed to fully utilize the power of Big Data and Distributed Computing System like NYU HPC Greene.

Currently, only equirectangular images are supported.

-----

To fully utilize the power of Multiple GPUs, the project is designed under the philosophy of 
 the so-called ["Actor Model"](https://en.wikipedia.org/wiki/Actor_model), where each computing actor only has access to its own data, and the messages (throw Websockets) amoung others. Currently, the logic is implemented in a proof-of-concept and naive way, which will be upgraded with some industrial-level infrastructures in the future (e.g. using [Ray](https://github.com/ray-project/ray)).

## Roadmap

- [x] Multi-Processes Actor Model (Proof-of-Concept)
- [x] Graph Database (Proof-of-Concept)
- [x] Nuxt.js & Vue.js & THREE.js Visualization (Proof-of-Concept)
- [x] Naive Bundle Adjustment with Gradient-based Optimization
- [ ] Pose Graph Optimization
- [ ] Loop Closure Detection
- [ ] Semantic SLAM, Global/Local Map Optimization, Deep Learning-based Methods, etc.
- [ ] Parallel-And-Distributed of all the above

## License

`torchslam` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.
