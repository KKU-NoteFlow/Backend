import httpx

GOOGLE_USER_INFO_URL = "https://www.googleapis.com/oauth2/v2/userinfo"

async def get_google_user_info(access_token: str) -> dict:
    headers = {"Authorization": f"Bearer {access_token}"}
    async with httpx.AsyncClient() as client:
        res = await client.get(GOOGLE_USER_INFO_URL, headers=headers)
        res.raise_for_status()
        return res.json()
