# Agent3_260324.py 전략 계획

## 1. 개요

### 목적
DB가 없을 때 Agent 2에서 추출한 제약(constraints)을 기반으로 component 조합을 통해 적절한 MOF를 제안

### 문제 정의
- **기존流程**: Agent 2 → Matchmaker → DB에서 매칭 → 결과
- **새로운流程**: Agent 2 → Agent 3 (mof2zeo 예측) → 순위 매기기 → MOF 제안

### 핵심 아이디어
```
Agent 2 constraints → 가능한 모든 (topology, node, edge) 조합 생성 
                    → mof2zeo 모델로 각 조합의 기하학 예측
                    → Agent 2의 target geometry와 비교
                    → 순위 매기기 → 상위 N개 MOF 제안
```

---

## 2. 입력/출력 정의

### 입력 (Agent 2의 출력)
```json
{
  "node_query": {
    "metals_include": ["Zr", "Hf"],
    "connectivity": [12],
    "nuclearity": 6,
    "abstract_features": {"has_open_metal_site": true}
  },
  "linker_query": {
    "connectivity": 2,
    "length_min": 8.0,
    "length_max": 15.0,
    "is_rigid": true,
    "functional_groups": ["Carboxyl"]
  },
  "geometry_filter": {
    "target_Di_min": 12.0,
    "target_Di_max": 20.0,
    "target_Df_min": 7.0,
    "target_Df_max": 10.0,
    "target_sa_min": 1000.0,
    "target_vf_min": 0.5
  }
}
```

### 출력
```json
{
  "ranked_mofs": [
    {
      "rank": 1,
      "topology": "fcu",
      "node": "N164",
      "edge": "E70",
      "predicted_geometry": {
        "di": 15.2,
        "df": 8.5,
        "sa": 2500.0,
        "vf": 0.65,
        "density": 1.2
      },
      "match_score": 0.85,
      "geometry_match": {
        "di": "✓ (12-20范围内)",
        "df": "✓ (7-10范围内)",
        "sa": "✓ (1000以上)",
        "vf": "✓ (0.5以上)"
      }
    },
    ...
  ],
  "summary": {
    "total_combinations": 500,
    "valid_after_geometry_filter": 45,
    "target_metric": "H2 storage"
  }
}
```

---

## 3. 컴포넌트 조합 생성 전략

### 3.1 유효한 topology 찾기
```python
# node_query에서 연결성 추출
node_cn = constraints['node_query']['connectivity']  # 예: [12]
linker_cn = constraints['linker_query']['connectivity']  # 예: 2

# topology.txt에서 조건 충족하는 토폴로지만 선택
# 조건: node_cn을 포함하고, edge_cn == linker_cn
valid_topologies = []
for topo in topology_list:
    if topo.node_cn in node_cn and topo.edge_cn == linker_cn:
        valid_topologies.append(topo)
```

### 3.2 유효한 node 찾기
```python
# metals_include 필터
valid_nodes = []
for node in node_list:
    if node.metals in constraints['node_query']['metals_include']:
        if node.connectivity in constraints['node_query']['connectivity']:
            valid_nodes.append(node)

# nuclearity 필터 (있는 경우)
if 'nuclearity' in constraints['node_query']:
    valid_nodes = [n for n in valid_nodes if n.nuclearity == req_nuclearity]
```

### 3.3 유효한 edge 찾기
```python
# linker_query 기반 필터링
valid_edges = []
for edge in edge_list:
    if edge.connectivity == constraints['linker_query']['connectivity']:
        # length 범위
        if constraints['linker_query'].get('length_min'):
            if edge.length < constraints['linker_query']['length_min']: continue
        if constraints['linker_query'].get('length_max'):
            if edge.length > constraints['linker_query']['length_max']: continue
        # rigidity 필터
        if constraints['linker_query'].get('is_rigid') is not None:
            if edge.is_rigid != constraints['linker_query']['is_rigid']: continue
        valid_edges.append(edge)
```

### 3.4 조합 생성
```python
# Cartesian product로 모든 조합 생성
all_combinations = []
for topo in valid_topologies:
    for node in valid_nodes:
        for edge in valid_edges:
            # 추가 필터: topology가 해당 node/edge를 지원하는지 확인
            if is_compatible(topo, node, edge):
                all_combinations.append((topo, node, edge))
```

---

## 4. mof2zeo 모델 연동

### 4.1 모델 구조 분석
- **입력**: topology + node + edge (각각 one-hot encoding)
- **출력**: 7개 기하학 속성 (Di, Df, SA, VF, density, dif, cv)
- **설정**: latent_dim=128, hid_dim1=64, hid_dim2=32

### 4.2 예측 파이프라인
```python
import torch
from mof2zeo.model import MOFNET

def predict_geometry(topology, node, edge, model, topo_dict, node_dict, edge_dict):
    # ID → 인덱스 변환
    topo_idx = topo_dict.get(topology, 0)
    node_idx = node_dict.get(node, 0)
    edge_idx = edge_dict.get(edge, 0)
    
    # 텐서 생성
    mof_tensor = torch.tensor([[topo_idx, node_idx, edge_idx]])
    
    # 예측 (스케일러를 통한 역변환 필요)
    pred_scaled = model(mof_tensor)
    pred_geometry = scaler.decode(pred_scaled)
    
    return {
        "di": pred_geometry[0],
        "df": pred_geometry[1],
        "sa": pred_geometry[2],
        "vf": pred_geometry[3],
        "density": pred_geometry[4],
        "dif": pred_geometry[5],
        "cv": pred_geometry[6]
    }
```

---

## 5. 순위 매기기 알고리즘

### 5.1 매칭 점수 계산
```python
def calculate_match_score(predicted, target):
    score = 0.0
    weights = {"di": 0.25, "df": 0.25, "sa": 0.2, "vf": 0.2, "density": 0.1}
    
    for key, weight in weights.items():
        if key in target and target[key] is not None:
            pred_val = predicted[key]
            min_val = target.get(f"{key}_min", float('-inf'))
            max_val = target.get(f"{key}_max", float('inf'))
            
            if min_val <= pred_val <= max_val:
                # 범위 내: 최대 점수
                score += weight
            else:
                # 범위 밖: 거리 기반 감점
                if pred_val < min_val:
                    penalty = (min_val - pred_val) / min_val
                else:
                    penalty = (pred_val - max_val) / max_val
                score += weight * (1 - min(penalty, 1.0))
    
    return score
```

### 5.2 정렬 및 필터링
```python
# 1. 모든 조합에 대해 기하학 예측
results = []
for combo in all_combinations:
    pred_geo = predict_geometry(combo, model, ...)
    score = calculate_match_score(pred_geo, geometry_filter)
    results.append({
        "combo": combo,
        "geometry": pred_geo,
        "score": score
    })

# 2. 점수 순으로 정렬
results.sort(key=lambda x: x["score"], reverse=True)

# 3. 상위 N개 선택 (예: 상위 20개)
top_mofs = results[:20]
```

---

## 6. 구현 구조

### Agent3_260324.py 구성

```python
# =============================================================================
# Agent 3: MOF Generator with Geometry Prediction
# =============================================================================

class GeometryPredictor:
    """mof2zeo 모델을 사용하여 topology+node+edge → geometry 예측"""
    
    def __init__(self, model_path, config):
        # 모델 로드 및 스케일러 초기화
        pass
    
    def predict(self, topology, node, edge):
        # 단일 조합에 대한 기하학 예측
        pass

class ComponentGenerator:
    """Agent 2의 제약을 기반으로 유효한 컴포넌트 조합 생성"""
    
    def generate_combinations(self, constraints):
        # topology, node, edge 필터링 및 조합 생성
        pass

class MOFRanker:
    """예측된 기하학과 목표 기하학 비교하여 순위 매기기"""
    
    def rank(self, combinations, predicted_geometries, target_geometry):
        # 점수 계산 및 정렬
        pass

class Agent3Handler:
    """Main Agent 3 Handler - 위 클래스들을 조율"""
    
    def __init__(self):
        self.predictor = GeometryPredictor(...)
        self.generator = ComponentGenerator(...)
        self.ranker = MOFRanker(...)
    
    def generate_mof_proposals(self, constraints, top_n=20):
        """주어진 제약에 대한 상위 N개 MOF 제안"""
        # 1. 컴포넌트 조합 생성
        # 2. 각 조합의 기하학 예측
        # 3. 순위 매기기
        # 4. 결과 반환
        pass
```

---

## 7. 의존성 및 설정

### 필요한 파일
1. **mof2zeo 모델**: `/home/users/seunghh/_hd1/autollm/260225_mof2zeo/ckpt/epoch=478-step=213634.ckpt`
2. **설정 파일**: `/home/users/seunghh/_hd1/autollm/mof2zeo/mof2zeo/config.yaml`
3. **사전 파일**: topology.txt, node.txt, edge.txt
4. **스케일러**: mean, std CSV 파일

### Python 의존성
```python
import torch
import yaml
import pandas as np
from mof2zeo.model import MOFNET
from mof2zeo.dataset import Scaler
```

---

## 8. 데이터 흐름

```
┌─────────────────────────────────────────────────────────────────┐
│                         Agent 2 출력                             │
│  node_query + linker_query + geometry_filter                   │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    ComponentGenerator                            │
│  • metals_include → 유효한 node 필터링                          │
│  • connectivity → 유효한 topology 필터링                        │
│  • linker 조건 → 유효한 edge 필터링                             │
│  • 모든 조합 생성 (Cartesian product)                           │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   GeometryPredictor                             │
│  • 각 조합 (topo+node+edge) → mof2zeo 예측                     │
│  • Di, Df, SA, VF, density, dif, cv 예측                       │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                       MOFRanker                                 │
│  • target geometry와 비교                                       │
│  • 매칭 점수 계산                                               │
│  • 점수 순 정렬                                                 │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Agent 3 출력                                │
│  • ranked_mofs: 상위 N개 제안                                   │
│  • summary: 통계 정보                                          │
└─────────────────────────────────────────────────────────────────┘
```

---

## 9. 고려사항

### 9.1 성능 최적화
- 배치 처리: 여러 조합을 한 번에 예측
- GPU 활용: CUDA 가속화
- 캐싱: 자주 사용되는 조합의 예측 결과 저장

### 9.2 한계점
- mof2zeo 모델의 예측 정확도 зависит
- unseen 조합에 대한 일반화 한계
- 예측과 실제 inúmer의 차이

### 9.3 확장 가능성
- 다중 목표 최적화 (bandgap + H2 storage 동시 만족)
-不确定性 예측 (예측의 confidence interval)
- 다중 MOF 제안 시 ensemble 방식

---

## 10. 다음 단계

1. **Agent3_260324.py 구현 시작**
   - ComponentGenerator 구현
   - GeometryPredictor 구현  
   - MOFRanker 구현
   - Agent3Handler 통합

2. **테스트**
   - 단위 테스트 (개별 클래스)
   - 통합 테스트 (전체 파이프라인)
   - 실제 제약으로 검증