import time
from typing import Any, Dict, List


def completion_with_retry(
    model: str,
    messages: List[Dict[str, str]],
    temperature: float,
    max_retries: int = 5,
    base_delay: float = 2.0,
    **kwargs: Any,
) -> str:
    """모델을 호출하고 생성된 텍스트를 반환한다. 실패 시 지수 백오프로 재시도한다."""
    # litellm 은 설치가 무거워서, 실제 호출이 필요한 시점에만 불러온다
    # (덕분에 litellm 없이도 나머지 모듈을 임포트하고 테스트할 수 있다)
    from litellm import completion

    last_error = None

    for attempt in range(1, max_retries + 1):
        try:
            response = completion(model=model, messages=messages, temperature=temperature, **kwargs)
            return response.choices[0].message.content.strip()
        except Exception as error:  # 어떤 API 오류든 재시도 대상
            last_error = error
            if attempt == max_retries:
                break
            delay = base_delay * (2 ** (attempt - 1))
            print(f"  [재시도 {attempt}/{max_retries}] {model} 호출 실패: {error} -> {delay:.0f}초 후 재시도")
            time.sleep(delay)

    raise RuntimeError(f"{model} 호출이 {max_retries}회 모두 실패했습니다.") from last_error
