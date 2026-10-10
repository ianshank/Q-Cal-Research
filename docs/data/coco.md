---
name: COCO 2017
license: CC-BY-4.0
images_license: Flickr-Terms-of-Use
source: https://cocodataset.org/#download
used_for: in-domain calibrator fitting, threshold selection and evaluation (val2017 subsets)
---

# Dataset card: COCO 2017

- **Annotations** are licensed CC BY 4.0 by the COCO Consortium. **Images** remain under
  the Flickr Terms of Use of their owners. [Likely: plan Appendix E.4; the official terms
  page was unreachable from the sandbox.] Ian confirms this in the `DECISIONS.md` dataset
  allowlist.
- **Source.** Only the official download is used for paper numbers. A Hugging Face mirror
  without a license tag is acceptable for CI fixtures only. The smoke test needs neither,
  because it uses a synthetic fixture.
- **Location.** `data/raw/coco/` (DVC), never committed. `configs/lab.toml` points at
  `annotations/instances_val2017.json` and `val2017/`.
- **Splits.** The manifests in `data/manifests/` define the four Q-Cal splits as subsets of
  val2017. `qcal leakage` checks they are disjoint.
- **Kuzucu et al. splits.** The paper splits val2017 at random into minival and minitest
  (arXiv:2405.20459, Sec. 4.3). Their published membership comes from a CC BY-NC-SA 4.0
  repository. Whether to import it (`python -m qcal_lab splits import`) or to draw a fresh
  seeded partition (`python -m qcal_lab splits partition`) is Ian's decision, recorded in
  `DECISIONS.md`.
