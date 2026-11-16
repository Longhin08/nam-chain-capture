"""Gera packaging/icon.{png,ico,icns}: um pedal mostarda com dois knobs e footswitch."""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

S = 1024
HERE = Path(__file__).parent


def draw() -> Image.Image:
    img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    # sombra
    sh = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    ImageDraw.Draw(sh).rounded_rectangle((236, 120, 812, 960), 96, fill=(0, 0, 0, 150))
    img.alpha_composite(sh.filter(ImageFilter.GaussianBlur(28)))
    d = ImageDraw.Draw(img)
    # gabinete com degradê mostarda
    body = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    grad = Image.new("RGBA", (1, 840))
    top, bot = (0xE8, 0xBB, 0x42), (0xB9, 0x8B, 0x22)
    for y in range(840):
        t = y / 839
        grad.putpixel((0, y), tuple(int(a + (b - a) * t) for a, b in zip(top, bot)) + (255,))
    grad = grad.resize((560, 840))
    mask = Image.new("L", (560, 840), 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, 559, 839), 90, fill=255)
    body.paste(grad, (232, 96), mask)
    img.alpha_composite(body)
    # LED
    d.ellipse((300, 160, 348, 208), fill=(255, 90, 78))
    # knobs (ponteiros em ângulos diferentes, como ganhos ajustados)
    import math

    for cx, ang in ((395, -40), (629, 35)):
        cy = 350
        d.ellipse((cx - 88, cy - 88, cx + 88, cy + 88), fill=(29, 32, 38))
        d.ellipse((cx - 76, cy - 76, cx + 76, cy + 76), fill=(62, 67, 77))
        a = math.radians(ang - 90)
        x0, y0 = cx + 22 * math.cos(a), cy + 22 * math.sin(a)
        x1, y1 = cx + 66 * math.cos(a), cy + 66 * math.sin(a)
        d.line((x0, y0, x1, y1), fill=(237, 234, 226), width=18)
    # serigrafia: três estágios ligados por um cabo (a cadeia)
    d.rounded_rectangle((290, 562, 734, 574), 6, fill=(27, 29, 34))
    for i, cx in enumerate((340, 512, 684)):
        fill = [(27, 29, 34), (123, 184, 168), (27, 29, 34)][i]
        d.rounded_rectangle((cx - 44, 528, cx + 44, 608), 14, fill=fill)
    # footswitch
    d.ellipse((432, 690, 592, 850), fill=(62, 67, 75))
    d.ellipse((444, 696, 580, 832), fill=(200, 205, 212))
    d.ellipse((470, 716, 554, 800), fill=(236, 239, 242))
    return img


def main():
    img = draw()
    img.save(HERE / "icon.png")
    img.save(HERE / "icon.ico", sizes=[(s, s) for s in (16, 24, 32, 48, 64, 128, 256)])
    img.save(HERE / "icon.icns")


if __name__ == "__main__":
    main()
