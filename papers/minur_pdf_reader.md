
以下是官网的请求格式
import requests

token = "官网申请的api token"
url = "https://mineru.net/api/v4/extract/task"
header = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {token}"
}
data = {
    "url": "https://cdn-mineru.openxlab.org.cn/demo/example.pdf",
    "model_version": "vlm"
}

res = requests.post(url,headers=header,json=data)
print(res.status_code)
print(res.json())
print(res.json()["data"])

以下是我的token：
eyJ0eXBlIjoiSldUIiwiYWxnIjoiSFM1MTIifQ.eyJqdGkiOiI2ODAwMDcwNCIsInJvbCI6IlJPTEVfUkVHSVNURVIiLCJpc3MiOiJPcGVuWExhYiIsImlhdCI6MTc3OTg5MzMyOCwiY2xpZW50SWQiOiJsa3pkeDU3bnZ5MjJqa3BxOXgydyIsInBob25lIjoiMTM1NTM2OTMyODQiLCJvcGVuSWQiOm51bGwsInV1aWQiOiJkZDM5OTE1Yi0wNWQ2LTRjZTctOGE2OC05MTBkZDU0YTFmOWMiLCJlbWFpbCI6IiIsImV4cCI6MTc4NzY2OTMyOH0.q_9emw6ItJR02FKt8K3db6gxMF84dy5V43hN6vWCHfYw9h9w2w5EqRFJy_ETorlVZXfmusqEhXBVpzVrjqsCFQ