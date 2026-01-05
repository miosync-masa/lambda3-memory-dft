"""
Holographic Interpretation Module
=================================

DSEの履歴依存構造をAdS/CFT的に解釈するモジュール。

核心的アイデア:
    - φ_history (位相蓄積履歴) → Bulk geometry
    - 非マルコフ性 → Bulk の深さ方向 (z)
    - RT entropy → 履歴の複雑さ/エンタングルメント

physics/ が「材料科学としての解釈」を担当するのに対し、
holographic/ は「量子重力としての解釈」を担当する。

DSEのcore/solversは一切変更せず、「解釈層」として独立。

Usage:
    from memory_dft.holographic import HolographicDual, quick_holographic_analysis
    
    # 簡易解析
    results = quick_holographic_analysis(phi_history)
    
    # 詳細制御
    holo = HolographicDual(L_ads=1.0, Z_depth=16)
    bulk = holo.history_to_bulk(phi_history)
    S_RT = holo.rt_entropy()
    C_V = holo.complexity_volume()

Author: Masamichi Iizumi & Tamaki (Miosync, Inc.)
"""

from .dual import (
    HolographicDual,
    quick_holographic_analysis,
    # Causality analysis
    transfer_entropy,
    crosscorr_at_lags,
    spearman_corr,
    verify_duality,
    plot_duality_analysis,
)

__all__ = [
    'HolographicDual',
    'quick_holographic_analysis',
    'transfer_entropy',
    'crosscorr_at_lags',
    'spearman_corr',
    'verify_duality',
    'plot_duality_analysis',
]
