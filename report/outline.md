# Báo cáo 10–20 trang (solo) — outline map thẳng vào experiments

> Quy ước: mỗi section có experiment đối ứng. Không viết claim nào mà không có figure/table.

1. **Intro (1–1.5p):** cow-on-grass; ERM vs invariant learning; RQ: (a) reproduce gap, (b) khi nào IRM gãy, (c) diagnostic nào chứng minh model bỏ màu.
2. **Background & Theory (3–4p):** SCM `Digit → Image ← Color`; env formalism; ERM objective; IRM constrained → IRMv1 penalty + annealing; V-REx + GroupDRO; điều kiện thất bại `E > |Ψ_s|` (Rosenfeld), overparam (SparseIRM), DomainBed model-selection critique; liên hệ Fairness (worst-group ≈ equalized-odds thô) + XAI (saliency).
3. **Method (2p):** pipeline data→train→eval→viz; configs fast/base/heavy + budget giờ T4; seeds + protocol chọn model trung thực (train-domain val vs OOD val).
4. **Experiments (5–8p):**
   - 4.1 Reproduce: bảng ERM/IRM/REx train vs OOD test + gap (outputs/sweep).
   - 4.2 A1 lambda-sweep (chữ U ngược) — outputs/ablate.
   - 4.3 A2 backbone MLP vs CNN.
   - 4.4 A3 failure demos: single-env, patch-spurious-2, small-n.
   - 4.5 A4 cross-dataset FashionMNIST.
   - 4.6 Diagnostics: reliability/ECE, saliency ERM-vs-IRM, probe decodability.
5. **Discussion & Limitations (1–2p):** IRM thắng synthetic nhưng yếu trên real (Waterbirds bỏ qua vì budget); benchmark-disagreement (ICLR'25); seed variance.
6. **Conclusion (0.5p) + Ethics note:** spurious = bias; deploy risk.
7. **Appendix:** configs, số liệu full seeds, hướng reproduce (`uv sync`, 3 lệnh).

**Demo live (5–7 phút):** chạy `configs/fast.yaml` (~5 phút T4) → show acc_bar + saliency + 1 failure case (single-env).
