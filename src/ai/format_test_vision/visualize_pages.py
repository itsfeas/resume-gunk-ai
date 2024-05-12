from typing import Iterable, Any
import pdfminer.layout as pdf
from pdfminer.high_level import extract_pages, LAParams

def show_ltitem_hierarchy(o: Any, depth=0):
    """Show location and text of LTItem and all its descendants"""
    if depth == 0:
        print('element                        stroking color  text')
        print('------------------------------ --------------  ----------')

    print(
        f'{get_indented_name(o, depth):<30.30s} '
        f'{get_optional_fontinfo(o):<20.20s} '
        f'{get_optional_color(o):<17.17s}'
        f'{get_optional_text(o)}'
    )

    if isinstance(o, Iterable):
        for i in o:
            # if not isinstance(i, pdf.LTChar) and not isinstance(i, pdf.LTAnno):
            #     show_ltitem_hierarchy(i, depth=depth + 1)
            show_ltitem_hierarchy(i, depth=depth + 1)


def get_indented_name(o: Any, depth: int) -> str:
    """Indented name of class"""
    return '  ' * depth + o.__class__.__name__


def get_optional_fontinfo(o: Any) -> str:
    """Font info of LTChar if available, otherwise empty string"""
    if hasattr(o, 'fontname') and hasattr(o, 'size'):
        return f'{o.fontname} {round(o.size)}pt'
    return ''

def get_optional_color(o: Any) -> str:
    """Font info of LTChar if available, otherwise empty string"""
    if hasattr(o, 'graphicstate'):
        return f'{o.graphicstate.scolor}'
    return ''


def get_optional_text(o: Any) -> str:
    """Text of LTItem if available, otherwise empty string"""
    if hasattr(o, 'get_text'):
        return o.get_text().strip()
    return ''

if __name__ == "__main__":
    pdf_path = "./dataset/"
    img_path = "./dataset/img/"
    file_name = "resume13.pdf"
    params = LAParams(char_margin=200, line_margin=1)
    pages = extract_pages(pdf_path+file_name, laparams=params)
    show_ltitem_hierarchy(pages)
