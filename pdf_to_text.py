# patent_easyocr_simple.py

import easyocr
import numpy as np
from PIL import Image
import re
from typing import List, Tuple
from pdf2image import convert_from_path


class PatentEasyOCRProcessor:
    """
    Простая версия без layoutparser:
    • PDF → изображения через poppler/pdf2image
    • OCR через EasyOCR
    • Сборка строк по bbox
    • Извлечение раздела (57) Реферат
    """

    def __init__(self, lang: str = "ru", dpi: int = 300, poppler_path: str = None):
        self.reader = easyocr.Reader([lang], gpu=False)
        self.dpi = dpi
        self.poppler_path = poppler_path

    def _pdf_to_arrays(self, pdf_path: str) -> List[np.ndarray]:
        pages = convert_from_path(pdf_path, dpi=self.dpi, poppler_path=self.poppler_path)
        return [np.array(p.convert("RGB")) for p in pages]

    def _image_to_array(self, img_path: str) -> np.ndarray:
        return np.array(Image.open(img_path).convert("RGB"))

    def _reconstruct_text(self, ocr_data: List[Tuple[List[List[float]], str, float]]) -> str:
        # сгруппировать все фрагменты в строки по близости Y
        lines: List[Tuple[float, List[Tuple[float, str]]]] = []
        for box, txt, _ in ocr_data:
            ys = [p[1] for p in box]; xs = [p[0] for p in box]
            y_mid, x_min = sum(ys)/4, min(xs)
            placed = False
            for i, (y0, segs) in enumerate(lines):
                if abs(y_mid - y0) < 15:  # порог «в одной строке»
                    segs.append((x_min, txt)); placed = True
                    # обновить средний Y
                    lines[i] = ((y0*len(segs)+y_mid)/(len(segs)+1), segs)
                    break
            if not placed:
                lines.append((y_mid, [(x_min, txt)]))

        # собрать текст
        lines.sort(key=lambda x: x[0])
        out = []
        for _, segs in lines:
            segs.sort(key=lambda s: s[0])
            out.append(" ".join(w for _, w in segs))
        return "\n".join(out)

    def _extract_abstract(self, text: str) -> str:
        m = re.search(r"\(57\)", text)
        if not m:
            return ""
        start = m.end()
        mr = re.search(r"реферат\s*[:\-]?", text[start:], flags=re.IGNORECASE)
        if mr:
            start += mr.end()
        m2 = re.search(r"\(\d{2}\)", text[start:])
        end = (m2.start()+start) if m2 else len(text)
        return text[start:end].strip()

    def process_files(self, paths: List[str]) -> Tuple[str, str]:
        # собрать все bbox‑фрагменты
        fragments: List[Tuple[List[List[float]], str, float]] = []
        for p in paths:
            ext = p.lower().rsplit(".",1)[-1]
            imgs = self._pdf_to_arrays(p) if ext=="pdf" else [self._image_to_array(p)]
            for img in imgs:
                # detail=1 — bbox, текст, confidence
                fragments.extend(self.reader.readtext(img, detail=1))

        full_text = self._reconstruct_text(fragments)
        abstract  = self._extract_abstract(full_text)
        return full_text, abstract


if __name__ == "__main__":
    # пример
    processor = PatentEasyOCRProcessor(lang="ru", dpi=300, poppler_path=None)
    files = ["test1.pdf"]
    full, ref57 = processor.process_files(files)

    # print("=== ПОЛНЫЙ ТЕКСТ ===")
    # print(full)
    print("\n=== РЕФЕРАТ (57) ===")
    print(ref57)
