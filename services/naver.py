import httpx

NAVER_USER_INFO_URL = "https://openapi.naver.com/v1/nid/me"

async def get_naver_user_info(access_token: str) -> dict:
    headers = {"Authorization": f"Bearer {access_token}"}
    async with httpx.AsyncClient() as client:
        res = await client.get(NAVER_USER_INFO_URL, headers=headers)
        res.raise_for_status()
        data = res.json()
        return data.get("response", {})
