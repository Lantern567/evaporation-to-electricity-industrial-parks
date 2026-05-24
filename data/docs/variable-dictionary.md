# Variable Dictionary

本文档记录研究方案中的核心变量、构造方法和预期解释。

## Core Variables

| 变量 | 含义 | 构造方法 | 预期方向 |
|---|---|---|---|
| `EDI` | 蒸发亏缺指数 | `(基准期蒸发量 - 当期蒸发量) / 基准期蒸发量` | 越高表示自然消纳能力越弱 |
| `H_s` | 行业高盐暴露度 | 行业浓盐水系数乘以自然蒸发依赖比例 | 越高，受蒸发冲击越大 |
| `NES` | 自然消纳替代强度 | 缺失自然蒸发量转化为机械处理量的比例 | 决定电驱负荷增量 |
| `DeltaV` | 缺失自然蒸发水量 | 浓盐水量乘以自然蒸发依赖度和蒸发亏缺 | 用于工程放大 |
| `DeltaElec` | 额外用电量 | `DeltaV × 单位处理电耗` | 带来成本和碳排变化 |
| `DeltaSteam` | 额外蒸汽或余热需求 | `DeltaV × 单位热耗` | 影响能源成本和余热配置 |
| `R_star` | 最低冗余率 | 满足可靠性和碳约束的最小绿电或设备冗余 | 用于园区规划建议 |

## Outcome Variables

| 变量 | 说明 |
|---|---|
| `treatment_electricity` | 企业或园区污水处理相关用电量 |
| `treatment_cost` | 电力、蒸汽、药剂、设备折旧、盐渣处置和土地等综合成本 |
| `carbon_emissions` | 水处理新增电力和热力带来的碳排 |
| `pond_expansion_capex` | 扩建蒸发池、防渗、土建和泵送系统投资 |
| `zld_operation_intensity` | 零排放设备运行强度或负荷率 |
| `park_design_failure` | 绿电匹配率、碳强度、峰值负荷或库容约束任一失效的指示变量 |

## Engineering Parameters

| 参数 | 说明 |
|---|---|
| `SEC_mvr` | MVR 单位处理电耗 |
| `SEC_membrane` | 膜浓缩单位处理电耗 |
| `SEC_crystallization` | 蒸发结晶单位处理电耗 |
| `EF_grid` | 当前电网排放因子 |
| `EF_green` | 绿电边界下排放因子 |
| `land_cost` | 蒸发池扩建土地和防渗成本 |
| `solid_waste_cost` | 盐渣处置和运输成本 |
| `heat_recovery_rate` | 余热替代率 |

