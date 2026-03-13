from tritonclient.utils import *
import tritonclient.grpc as grpcclient
server_url = "example-server.local:8081"
with grpcclient.InferenceServerClient(
    url=server_url,
    verbose=False,
) as client:
    if not client.is_server_live():
        raise Exception("Triton 서버가 Live 상태가 아닙니다.")
    if not client.is_server_ready():
        raise Exception("Triton 서버가 Ready 상태가 아닙니다.")
    print(client.is_server_live())
check=True