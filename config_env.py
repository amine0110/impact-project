import gdown

gdown.download(
    "https://drive.google.com/uc?id=1f-jEh05PBaHXLRwBvrYgSSPtYLuBeiPm",
    use_cookies=False,
    output="human_falling_detect_tracks/Models/yolo-tiny-onecls/yolov3-tiny-onecls.cfg",
)

gdown.download(
    "https://drive.google.com/uc?id=1GbH6UQpBvOwLO_gozaH9QFnL23j8plPb",
    use_cookies=False,
    output="human_falling_detect_tracks/Models/yolo-tiny-onecls/best-model.pth",
)

gdown.download(
    "https://drive.google.com/uc?id=1m5tIKH8INJ82GI7q6tWtPPDJ55jgpKfg",
    use_cookies=False,
    output="human_falling_detect_tracks/Models/sppe/fast_res50_256x192.pth",
)

gdown.download(
    "https://drive.google.com/uc?id=1MAk2v0peCa4Fs9zw9-N3ssuG4Y1MIItO",
    use_cookies=False,
    output="human_falling_detect_tracks/Models/TSSTG/tsstg-model.pth",
)
