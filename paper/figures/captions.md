# RIIC-ReID Main-Paper Figure Captions

## Figure 1
Retrieval-preferred correction is not equivalent to human-perceptual enhancement. The same ATRW query (query/21/001132.jpg, luminance 0.509) is processed by RetinexNet, Zero-DCE++, and RIIC-ReID. Although the perceptual baselines visibly alter brightness and contrast, they produce weaker ranked retrieval lists than RIIC-ReID. This supports the paper's central claim that illumination correction for retrieval should be optimized for the downstream embedding geometry rather than for human-perceptual appearance alone.

## Figure 2
Overview of RIIC-ReID. A bounded feasible correction stage first constructs a safe operating region using coarse illumination estimation, sensitivity modulation, and constrained inverse scaling. Trust-controlled adaptation then applies model-aware residual correction, identity protection, and stripe-wise rollback. The encoder keeps raw, base-corrected, and adapted branches and fuses them using stripe-aware branch attention. During training only, a frozen teacher provides manifold-tube, separation, and ranking supervision.

## Figure 3
Trust and geometry evidence for RIIC-ReID. Left: the trust modules generate interpretable spatial behavior, including identity protection, rollback control, correction-gap localization, illumination estimation, and color-risk detection. Right: in the teacher-centered projection, the RIIC-ReID query remains closer to the teacher-defined same-identity region than a perceptual baseline, illustrating the geometry-guided objective.
