# Data Scripts

本目录用于放置可复现的数据处理脚本。建议按以下顺序逐步创建：

1. `build_evaporation_deficit.py`: 读取气象数据，构建县域、企业和园区尺度 EDI。
2. `build_high_salinity_exposure.py`: 构建行业高盐暴露度和省份-行业-园区浓盐水矩阵。
3. `estimate_causal_effects.py`: 运行双向固定效应、事件研究和稳健性检验。
4. `convert_gap_to_energy_cost.py`: 将蒸发缺口转换为电力、蒸汽、设备、土地、成本和碳排。
5. `run_park_stress_test.py`: 计算零碳园区源荷匹配、设计失效概率和最低冗余率。
6. `optimize_provincial_allocation.py`: 运行省级设备-电力-热-储协同配置模型。

