from typing import List, Dict, Any, Optional
import json
import time
import logging
import requests
from src.data.schema import DomainType, PageType, QueryType, QueryItem
from src.data.query_sanitizer import QuerySanitizer
from src.config import LLMConfig

logger = logging.getLogger(__name__)

VIETNAMESE_PROMPT_TEMPLATES = {
    DomainType.LEGAL: """
Bạn là chuyên gia thẩm định tài liệu pháp lý Việt Nam.
Hãy đọc nội dung trang tài liệu (Luật, Nghị định, Thông tư, Quyết định, Tài liệu tổng luận) dưới đây và sinh ra 2 đến 3 câu hỏi tìm kiếm thực tế của người dùng:

Yêu cầu nghiêm ngặt:
1. Câu hỏi phải tự đủ nghĩa (stand-alone), người đọc bình thường không thấy trang này vẫn hiểu được.
2. TUYỆT ĐỐI KHÔNG dùng các từ chỉ vị trí/ngữ cảnh như: "trang này", "hình trên", "bảng dưới", "theo văn bản này", "trong tài liệu".
3. Câu hỏi tra cứu điều khoản, mức phạt, hoặc quy trình thủ tục thực tế.

Trả về duy nhất định dạng JSON (không thêm markdown ngoài khối json):
[
  {"query_text": "...", "query_type": "legal_clause", "hardness_level": "medium"},
  {"query_text": "...", "query_type": "fact_lookup", "hardness_level": "easy"}
]
""",
    DomainType.FINANCIAL: """
Bạn là chuyên gia phân tích báo cáo tài chính và dữ liệu kinh tế Việt Nam.
Hãy đọc nội dung trang tài liệu (Bảng cân đối, Báo cáo tài chính, Biểu mẫu, Số liệu thống kê) và sinh ra 2 đến 3 câu hỏi tìm kiếm:

Yêu cầu nghiêm ngặt:
1. Nêu rõ tên chỉ tiêu, đơn vị thực thể hoặc kỳ kế toán/năm.
2. TUYỆT ĐỐI KHÔNG dùng từ: "bảng này", "trang này", "bảng số liệu trên".
3. Truy vấn số liệu bảng hoặc so sánh chỉ số.

Trả về duy nhất định dạng JSON:
[
  {"query_text": "...", "query_type": "numeric_table", "hardness_level": "medium"},
  {"query_text": "...", "query_type": "multi_cell_comparison", "hardness_level": "hard"}
]
""",
    DomainType.EDUCATION: """
Bạn là chuyên gia tra cứu tài liệu khoa học, giáo trình đại học và công nghệ thông tin.
Hãy đọc nội dung trang giáo trình/sách kỹ thuật dưới đây và sinh ra 2 đến 3 câu hỏi tìm kiếm:

Yêu cầu nghiêm ngặt:
1. Nêu rõ khái niệm, thuật toán, định lý, kiến trúc hệ thống, hoặc câu lệnh cần tìm kiếm.
2. TUYỆT ĐỐI KHÔNG dùng từ: "trang này", "hình trên", "sơ đồ này", "đoạn trên".
3. Câu hỏi tự nhiên như sinh viên/kỹ sư đang tra cứu kiến thức.

Trả về duy nhất định dạng JSON:
[
  {"query_text": "...", "query_type": "fact_lookup", "hardness_level": "medium"},
  {"query_text": "...", "query_type": "paraphrase_or_abbreviation", "hardness_level": "hard"}
]
""",
    DomainType.HEALTHCARE: """
Bạn là chuyên gia tra cứu y khoa và dược phẩm.
Hãy đọc trang tài liệu y tế và sinh ra 2 đến 3 câu hỏi:

Yêu cầu nghiêm ngặt:
1. Nêu rõ tên thuốc, bệnh lý, triệu chứng hoặc phác đồ.
2. TUYỆT ĐỐI KHÔNG dùng từ: "trang này", "hình này", "bảng sau".

Trả về duy nhất định dạng JSON:
[
  {"query_text": "...", "query_type": "fact_lookup", "hardness_level": "easy"},
  {"query_text": "...", "query_type": "paraphrase_or_abbreviation", "hardness_level": "hard"}
]
""",
    DomainType.INFOGRAPHIC: """
Bạn là chuyên gia thị giác thông tin.
Hãy đọc trang biểu đồ / infographic và sinh ra 2 câu hỏi tìm kiếm:

Yêu cầu nghiêm ngặt:
1. Nêu rõ chủ đề biểu đồ, năm, hoặc địa phương cần so sánh.
2. TUYỆT ĐỐI KHÔNG dùng từ: "biểu đồ này", "ảnh trên", "hình bên".

Trả về duy nhất định dạng JSON:
[
  {"query_text": "...", "query_type": "chart_interpretation", "hardness_level": "medium"}
]
""",
}

class QueryGenerator:
    """
    Constructs prompt payloads, calls LLM APIs (OpenAI or Gemini),
    and parses outputs into validated QueryItem objects.
    """
    def __init__(
        self,
        sanitizer: Optional[QuerySanitizer] = None,
        llm_config: Optional[LLMConfig] = None,
    ):
        self.sanitizer = sanitizer or QuerySanitizer()
        self.llm_config = llm_config or LLMConfig()

    def get_prompt_for_page(
        self,
        domain: DomainType,
        page_text: str,
        page_num: int,
        doc_id: str,
    ) -> str:
        base_template = VIETNAMESE_PROMPT_TEMPLATES.get(
            domain, VIETNAMESE_PROMPT_TEMPLATES[DomainType.EDUCATION]
        )
        return f"{base_template}\n\n[NỘI DUNG VĂN BẢN TRANG {page_num} - TÀI LIỆU {doc_id}]:\n{page_text[:3000]}"

    def call_openai_api(self, prompt: str) -> Optional[str]:
        """Calls OpenAI Chat Completion API."""
        if not self.llm_config.openai_api_key:
            return None

        try:
            from openai import OpenAI
            client = OpenAI(api_key=self.llm_config.openai_api_key)
            response = client.chat.completions.create(
                model=self.llm_config.openai_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a professional Information Retrieval benchmark curator. Respond strictly with valid JSON.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=self.llm_config.temperature,
                response_format={"type": "json_object"} if "gpt-4" in self.llm_config.openai_model else None,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.warning(f"OpenAI API call failed (client): {e}")
            # Try raw request fallback
            headers = {
                "Authorization": f"Bearer {self.llm_config.openai_api_key}",
                "Content-Type": "application/json",
            }
            payload = {
                "model": self.llm_config.openai_model,
                "messages": [
                    {"role": "system", "content": "Respond strictly with valid JSON."},
                    {"role": "user", "content": prompt},
                ],
                "temperature": self.llm_config.temperature,
            }
            try:
                res = requests.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=30,
                )
                if res.status_code == 200:
                    data = res.json()
                    return data["choices"][0]["message"]["content"]
                else:
                    logger.warning(f"OpenAI API fallback failed with status {res.status_code}: {res.text}")
            except Exception as e2:
                logger.warning(f"OpenAI API fallback request failed: {e2}")
        return None

    def call_gemini_api(self, prompt: str) -> Optional[str]:
        """Calls Google Gemini API."""
        if not self.llm_config.gemini_api_key:
            return None

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.llm_config.gemini_model}:generateContent?key={self.llm_config.gemini_api_key}"
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": self.llm_config.temperature},
        }
        try:
            res = requests.post(url, json=payload, timeout=30)
            if res.status_code == 200:
                data = res.json()
                candidates = data.get("candidates", [])
                if candidates:
                    parts = candidates[0].get("content", {}).get("parts", [])
                    if parts:
                        return parts[0].get("text", "")
            else:
                logger.warning(f"Gemini API failed with status {res.status_code}: {res.text}")
        except Exception as e:
            logger.warning(f"Gemini API request failed: {e}")
        return None

    def generate_queries_for_page(
        self,
        domain: DomainType,
        page_text: str,
        page_num: int,
        doc_id: str,
        target_page_id: str,
        query_id_prefix: str = "q",
    ) -> List[QueryItem]:
        """
        Generates queries using the configured LLM API (OpenAI or Gemini),
        with fallback to heuristic generation if API fails.
        """
        if not page_text or len(page_text.strip()) < 30:
            return []

        prompt = self.get_prompt_for_page(domain, page_text, page_num, doc_id)
        raw_response = None

        if self.llm_config.provider == "openai" and self.llm_config.openai_api_key:
            raw_response = self.call_openai_api(prompt)

        if not raw_response and self.llm_config.gemini_api_key:
            raw_response = self.call_gemini_api(prompt)

        if raw_response:
            items = self.parse_and_validate_response(
                raw_response=raw_response,
                target_page_id=target_page_id,
                domain=domain,
                document_text=page_text,
                query_id_prefix=query_id_prefix,
            )
            if items:
                return items

        # Fallback: heuristic extraction from page text
        lines = [l.strip() for l in page_text.split("\n") if len(l.strip()) > 25 and len(l.strip()) < 150]
        fallback_items = []
        for idx, line in enumerate(lines[:4]):
            cleaned = self.sanitizer.clean_deictic_words(line)
            is_valid, reason = self.sanitizer.validate_query(cleaned)
            if is_valid:
                q_id = f"{query_id_prefix}_{target_page_id}_{idx+1:02d}"
                fallback_items.append(
                    QueryItem(
                        query_id=q_id,
                        query_text=cleaned,
                        domain=domain,
                        query_type=QueryType.FACT_LOOKUP,
                        source="heuristic_fallback",
                        target_page_ids=[target_page_id],
                        hardness_level="medium",
                        metadata={"validation_status": reason},
                    )
                )
        return fallback_items

    def parse_and_validate_response(
        self,
        raw_response: str,
        target_page_id: str,
        domain: DomainType,
        document_text: str = "",
        query_id_prefix: str = "q",
    ) -> List[QueryItem]:
        """Parses JSON response and validates each query."""
        cleaned_response = raw_response.strip()
        if "```json" in cleaned_response:
            cleaned_response = cleaned_response.split("```json")[1].split("```")[0].strip()
        elif "```" in cleaned_response:
            cleaned_response = cleaned_response.split("```")[1].split("```")[0].strip()

        try:
            parsed_data = json.loads(cleaned_response)
        except Exception as e:
            logger.warning(f"Failed to parse LLM response as JSON: {e}")
            logger.debug(f"Raw response: {raw_response[:500]}")
            return []

        if isinstance(parsed_data, dict):
            # Sometimes LLMs wrap array in {"queries": [...]}
            for key in ["queries", "questions", "data", "results"]:
                if key in parsed_data and isinstance(parsed_data[key], list):
                    parsed_data = parsed_data[key]
                    break

        if not isinstance(parsed_data, list):
            return []

        valid_items: List[QueryItem] = []
        for idx, item in enumerate(parsed_data):
            if not isinstance(item, dict):
                continue
            query_text = item.get("query_text", item.get("query", item.get("question", ""))).strip()
            query_text = self.sanitizer.clean_deictic_words(query_text)
            is_valid, reason = self.sanitizer.validate_query(query_text, document_text)
            if not is_valid:
                continue

            q_type_str = item.get("query_type", "fact_lookup")
            try:
                q_type = QueryType(q_type_str)
            except ValueError:
                q_type = QueryType.FACT_LOOKUP

            hardness = item.get("hardness_level", "medium")
            query_id = f"{query_id_prefix}_{target_page_id}_{idx+1:02d}"

            query_item = QueryItem(
                query_id=query_id,
                query_text=query_text,
                domain=domain,
                query_type=q_type,
                source="llm_assisted",
                target_page_ids=[target_page_id],
                hardness_level=hardness,
                metadata={"validation_status": reason},
            )
            valid_items.append(query_item)

        return valid_items
