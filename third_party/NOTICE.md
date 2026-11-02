# Inspected upstream conventions and licences

The Arithmetic pilot is a reconstruction from the primary paper and the pinned
sources below. No upstream model module, pretrained weight, FlashAttention
implementation or CUDA extension is imported. The inaccessible recipe code and
its software licence remain unverified; the paper's CC BY 4.0 article licence
does not establish a recipe software licence.

| Upstream | Pinned revision | Use | Retained notice |
|---|---|---|---|
| [sapientinc/HRM](https://github.com/sapientinc/HRM/tree/ac15626f8db096a63c775b84c9dc868776a6feda) | `ac15626f8db096a63c775b84c9dc868776a6feda` | Inspected layer dimensions, FF rounding, stablemax loss, carry/update and optimizer conventions. Handwritten eager model differs from its parameter-free RMSNorm and initial-state buffers. | [Apache 2.0](HRM_APACHE_LICENSE) |
| [SamsungSAILMontreal/TinyRecursiveModels](https://github.com/SamsungSAILMontreal/TinyRecursiveModels/tree/c01103738605ba39d1430519b1ee0c62f4c707f8) | `c01103738605ba39d1430519b1ee0c62f4c707f8` | Licence inspected; no TRM code or weights used in this implementation. | [MIT](TRM_MIT_LICENSE) |
| [lucidrains/adam-atan2-pytorch](https://github.com/lucidrains/adam-atan2-pytorch/tree/60ba5715f52d6b7f099bba1cba3baa1548e04410) | `60ba5715f52d6b7f099bba1cba3baa1548e04410` | Restricted handwritten dense update based on inspected MIT code: a=1.27, b=1, bias correction and LR-scaled decoupled decay. Optional cautious/regenerative features excluded. The HRM fused `adam_atan2` backend is different and was not installed. | [MIT](ADAM_ATAN2_LICENSE) |
