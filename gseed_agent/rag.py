from __future__ import annotations

import json
import re
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class GraphNode:
    """G-SEED 그래프의 단일 노드."""

    node_id: str
    node_type: str
    label: str
    payload: dict[str, Any]


class GSeedGraph:
    """G-SEED KB와 프로젝트 결과를 노드/엣지 그래프로 표현한다."""

    def __init__(self) -> None:
        self.nodes: dict[str, GraphNode] = {}
        self.edges: dict[str, list[tuple[str, str]]] = defaultdict(list)
        self.reverse_edges: dict[str, list[tuple[str, str]]] = defaultdict(list)

    def add_node(self, node_id: str, node_type: str, label: str, payload: dict[str, Any] | None = None) -> None:
        if node_id not in self.nodes:
            self.nodes[node_id] = GraphNode(node_id, node_type, label, payload or {})

    def add_edge(self, source: str, relation: str, target: str) -> None:
        self.edges[source].append((relation, target))
        self.reverse_edges[target].append((relation, source))

    def neighbors(self, node_id: str, max_hops: int = 1) -> list[dict[str, Any]]:
        """시작 노드 주변을 hop 단위로 탐색한다."""
        visited = {node_id}
        queue: deque[tuple[str, int]] = deque([(node_id, 0)])
        result: list[dict[str, Any]] = []

        while queue:
            current, depth = queue.popleft()
            if depth >= max_hops:
                continue

            for relation, target in self.edges.get(current, []):
                if target not in self.nodes:
                    continue
                result.append(
                    {
                        "source": self.nodes[current],
                        "relation": relation,
                        "target": self.nodes[target],
                        "hop": depth + 1,
                    }
                )
                if target not in visited:
                    visited.add(target)
                    queue.append((target, depth + 1))
        return result


def build_gseed_graph(kb: dict[str, Any], result: dict[str, Any] | None = None) -> GSeedGraph:
    """G-SEED KB와 현재 프로젝트 결과를 그래프로 변환한다."""
    graph = GSeedGraph()

    for domain in kb.get("domains", []):
        domain_id = f"domain:{domain.get('domain_no')}"
        graph.add_node(
            domain_id,
            "domain",
            str(domain.get("domain_name") or domain.get("domain_no")),
            {"domain_no": domain.get("domain_no"), "source_file": domain.get("source_file")},
        )

        for criterion in domain.get("criteria", []):
            criterion_id = f"criterion:{criterion.get('criterion_id')}"
            graph.add_node(
                criterion_id,
                "criterion",
                f"{criterion.get('criterion_id')} {criterion.get('name')}",
                criterion,
            )
            graph.add_edge(domain_id, "HAS_CRITERION", criterion_id)

            for variable in criterion.get("input_variables", []):
                name = variable.get("name")
                if not name:
                    continue
                variable_id = f"variable:{name}"
                graph.add_node(variable_id, "required_value", variable.get("label") or name, variable)
                graph.add_edge(criterion_id, "REQUIRES_VALUE", variable_id)

            for phase, documents in (criterion.get("required_documents") or {}).items():
                for document in documents or []:
                    doc_id = f"document:{_normalize_key(str(document))}"
                    graph.add_node(doc_id, "required_document", str(document), {"phase": phase})
                    graph.add_edge(criterion_id, "REQUIRES_DOCUMENT", doc_id)

            for rule in criterion.get("grade_rules", [])[:8]:
                rule_label = str(rule.get("original_text") or rule.get("condition") or rule.get("grade") or "")
                if not rule_label:
                    continue
                rule_id = f"rule:{criterion.get('criterion_id')}:{_normalize_key(rule_label)[:60]}"
                graph.add_node(rule_id, "grade_rule", rule_label, rule)
                graph.add_edge(criterion_id, "HAS_GRADE_RULE", rule_id)

    if result:
        _attach_project_result(graph, result)

    _attach_related_criteria(graph)
    return graph


class GSeedGraphRetriever:
    """항목-문서-변수-fact 관계를 따라가는 G-SEED 전용 그래프 검색기."""

    def __init__(self, kb: dict[str, Any], result: dict[str, Any] | None = None, max_hops: int = 2) -> None:
        self.kb = kb
        self.result = result
        self.max_hops = max_hops
        self.graph = build_gseed_graph(kb, result)

    def retrieve(self, query: str, limit: int = 4) -> dict[str, Any]:
        """질문에서 시작 노드를 찾고 주변 그래프 context를 반환한다."""
        start_nodes = self._find_start_nodes(query, limit=limit)
        paths: list[dict[str, Any]] = []
        for node_id in start_nodes:
            paths.extend(self.graph.neighbors(node_id, max_hops=self.max_hops))

        return {
            "query": query,
            "start_nodes": [self._node_to_dict(self.graph.nodes[node_id]) for node_id in start_nodes],
            "paths": [self._path_to_dict(path) for path in paths],
            "context_text": self.to_context_text(start_nodes, paths),
        }

    def to_context_text(self, start_nodes: list[str], paths: list[dict[str, Any]]) -> str:
        """LLM prompt에 넣을 수 있는 짧은 context 문장으로 변환한다."""
        lines: list[str] = []
        for node_id in start_nodes:
            node = self.graph.nodes[node_id]
            lines.append(f"[START] {node.node_type}: {node.label}")
            if node.node_type == "criterion":
                payload = node.payload
                if payload.get("purpose"):
                    lines.append(f"- 평가목적: {payload.get('purpose')}")
                if payload.get("method"):
                    lines.append(f"- 평가방법: {payload.get('method')}")

        grouped: dict[str, list[str]] = defaultdict(list)
        for path in paths:
            target = path["target"]
            grouped[path["relation"]].append(target.label)

        relation_labels = {
            "REQUIRES_DOCUMENT": "필요 문서",
            "REQUIRES_VALUE": "문서에서 확인할 값",
            "HAS_GRADE_RULE": "등급 기준",
            "HAS_EVIDENCE": "현재 프로젝트 근거",
            "HAS_STATUS": "현재 프로젝트 판정",
            "RELATED_TO": "관련 항목",
        }
        for relation, labels in grouped.items():
            title = relation_labels.get(relation, relation)
            unique_labels = _unique(labels)[:8]
            lines.append(f"- {title}: " + " / ".join(unique_labels))
        return "\n".join(lines)

    def as_langchain_retriever(self) -> Any:
        """LangChain BaseRetriever 호환 wrapper를 반환한다."""
        try:
            from langchain_core.documents import Document
            from langchain_core.retrievers import BaseRetriever
            from pydantic import ConfigDict, Field
        except ImportError as exc:  # pragma: no cover - 설치 환경 의존
            raise RuntimeError(
                "LangChain이 설치되어 있지 않습니다. "
                "`pip install -r requirements.txt` 실행 후 다시 시도해 주세요."
            ) from exc

        outer = self

        class _Retriever(BaseRetriever):
            model_config = ConfigDict(arbitrary_types_allowed=True)

            graph_retriever: GSeedGraphRetriever = Field(default=outer)

            def _get_relevant_documents(self, query: str) -> list[Document]:
                retrieved = self.graph_retriever.retrieve(query)
                return [
                    Document(
                        page_content=retrieved["context_text"],
                        metadata={
                            "start_nodes": retrieved["start_nodes"],
                            "paths": retrieved["paths"],
                            "retriever": "GSeedGraphRetriever",
                        },
                    )
                ]

        return _Retriever()

    def _find_start_nodes(self, query: str, limit: int) -> list[str]:
        normalized_query = query.lower()
        compact_query = re.sub(r"\s+", "", normalized_query)
        exact: list[str] = []

        # R-6.3 같은 항목 번호 직접 매칭.
        for match in re.finditer(r"r[-\s]?(\d+)\.(\d+)", normalized_query):
            criterion_id = f"criterion:R-{match.group(1)}.{match.group(2)}"
            if criterion_id in self.graph.nodes:
                exact.append(criterion_id)

        # 항목명 직접 매칭.
        for node in self.graph.nodes.values():
            if node.node_type != "criterion":
                continue
            name = str(node.payload.get("name") or "").lower()
            compact_name = re.sub(r"\s+", "", name)
            if compact_name and compact_name in compact_query and node.node_id not in exact:
                exact.append(node.node_id)

        if exact:
            return exact[:1]

        scored: list[tuple[int, str]] = []
        query_terms = [term for term in re.split(r"\s+", normalized_query) if term]
        for node in self.graph.nodes.values():
            if node.node_type not in {"criterion", "required_document", "required_value", "domain"}:
                continue
            haystack = json.dumps(
                {"label": node.label, "payload": node.payload},
                ensure_ascii=False,
            ).lower()
            score = sum(haystack.count(term) for term in query_terms)
            if score:
                scored.append((score, node.node_id))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [node_id for _, node_id in scored[:limit]]

    @staticmethod
    def _node_to_dict(node: GraphNode) -> dict[str, Any]:
        return {
            "node_id": node.node_id,
            "node_type": node.node_type,
            "label": node.label,
            "payload": node.payload,
        }

    def _path_to_dict(self, path: dict[str, Any]) -> dict[str, Any]:
        return {
            "source": self._node_to_dict(path["source"]),
            "relation": path["relation"],
            "target": self._node_to_dict(path["target"]),
            "hop": path["hop"],
        }


def _attach_project_result(graph: GSeedGraph, result: dict[str, Any]) -> None:
    """현재 프로젝트 평가 결과와 근거 fact를 그래프에 붙인다."""
    for row in result.get("criteria_results", []):
        criterion_node = f"criterion:{row.get('criterion_id')}"
        if criterion_node not in graph.nodes:
            continue

        status_id = f"status:{row.get('criterion_id')}"
        status_label = f"{row.get('status')} / {row.get('grade') or '-'} / {row.get('score') if row.get('score') is not None else '-'}점"
        graph.add_node(status_id, "project_status", status_label, row)
        graph.add_edge(criterion_node, "HAS_STATUS", status_id)

        for idx, evidence in enumerate(row.get("evidence", []) or []):
            fact_id = f"fact:{row.get('criterion_id')}:{idx}"
            fact_label = (
                f"{evidence.get('variable')}={evidence.get('value')} "
                f"({evidence.get('source_document')} p.{evidence.get('source_page')})"
            )
            graph.add_node(fact_id, "project_fact", fact_label, evidence)
            graph.add_edge(criterion_node, "HAS_EVIDENCE", fact_id)


def _attach_related_criteria(graph: GSeedGraph) -> None:
    """같은 전문분야의 인접 항목을 관련 항목으로 연결한다."""
    by_domain: dict[str, list[str]] = defaultdict(list)
    for node in graph.nodes.values():
        if node.node_type != "criterion":
            continue
        domain_no = str((node.payload.get("domain") or {}).get("번호") or "")
        if domain_no:
            by_domain[domain_no].append(node.node_id)

    for nodes in by_domain.values():
        sorted_nodes = sorted(nodes, key=_criterion_sort_key)
        for idx, node_id in enumerate(sorted_nodes):
            for neighbor in [idx - 1, idx + 1]:
                if 0 <= neighbor < len(sorted_nodes):
                    graph.add_edge(node_id, "RELATED_TO", sorted_nodes[neighbor])


def _criterion_sort_key(node_id: str) -> tuple[int, int, str]:
    match = re.search(r"R-(\d+)\.(\d+)", node_id)
    if match:
        return int(match.group(1)), int(match.group(2)), node_id
    match = re.search(r"R-ID-(\d+)", node_id)
    if match:
        return 99, int(match.group(1)), node_id
    return 100, 0, node_id


def _normalize_key(value: str) -> str:
    return re.sub(r"\W+", "_", value.strip().lower()).strip("_")


def _unique(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result
