import asyncio # 비동기화 모듈
from bleak import BleakScanner # BLE 검색 모듈

# 비동기 형태로 BLE 장치 검색
async def run():
    # 검색 시작 (검색이 종료될때까지 대기)
    # 기본 검색 시간은 5초이다.
    devices = await BleakScanner.discover()
    # 검색된 장치들 리스트 출력
    for d in devices:
        print(d)

# 비동기 이벤트 루프 생성
loop = asyncio.get_event_loop()
# 비동기 형태로 run(검색)함수 실행
# 완료될때까지 대기
loop.run_until_complete(run())