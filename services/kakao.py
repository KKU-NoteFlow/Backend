import httpx

KAKAO_USER_INFO_URL = "https://kapi.kakao.com/v2/user/me"

async def get_kakao_user_info(access_token: str) -> dict:
    headers = {"Authorization": f"Bearer {access_token}"}
    async with httpx.AsyncClient() as client:
        response = await client.get(KAKAO_USER_INFO_URL, headers=headers)
        response.raise_for_status()
        return response.json()