import mlx.core as mx

mx.random.seed(0)
cases = {
    "conv2d 8->8": (mx.conv2d, (64, 32, 32, 8), (8, 3, 3, 8)),
    "conv3d 3->24": (mx.conv3d, (4, 8, 16, 16, 3), (24, 3, 3, 3, 3)),
}
for name, (conv, xs, ws) in cases.items():
    x = mx.random.normal(xs)
    w = mx.random.normal(ws)
    ref = conv(x, w, padding=1, stream=mx.cpu)
    mx.eval(x, w, ref)
    bad = 0
    for _ in range(100):
        y = conv(x, w, padding=1)
        mx.async_eval(y)
        scalars = [mx.array(1000.0 + i) for i in range(64)]
        bad += int(not mx.allclose(y, ref, atol=1e-2).item())
    print(f"mlx {mx.__version__} {name}: {bad}/100 wrong", flush=True)
