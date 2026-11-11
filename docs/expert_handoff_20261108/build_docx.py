from pathlib import Path
import json,sys,hashlib
from docx import Document
from docx.shared import Cm,Pt,RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

root=Path(sys.argv[1]); output=Path(sys.argv[2])
data=json.loads((root/'content.json').read_text())
doc=Document()
sec=doc.sections[0]
sec.page_width=Cm(21);sec.page_height=Cm(29.7)
sec.top_margin=Cm(1.85);sec.bottom_margin=Cm(1.8)
sec.left_margin=Cm(2.0);sec.right_margin=Cm(2.0)
sec.header_distance=Cm(.75);sec.footer_distance=Cm(.75)
FONT='Songti SC'; SANS='PingFang SC'
def setfonts(run,size=11,bold=None,name=FONT):
    run.font.name='Arial';run.font.size=Pt(size);run.font.color.rgb=RGBColor(0,0,0)
    if bold is not None:run.bold=bold
    rp=run._element.get_or_add_rPr()
    rf=rp.rFonts
    if rf is None:rf=OxmlElement('w:rFonts');rp.insert(0,rf)
    for k,v in [('ascii','Arial'),('hAnsi','Arial'),('eastAsia',name),('cs','Arial')]:rf.set(qn('w:'+k),v)
    return run
for name in ['Normal','Title','Subtitle','Heading 1','Heading 2','Heading 3','Caption','List Bullet']:
    st=doc.styles[name]
    st.font.name='Arial';st.font.color.rgb=RGBColor(0,0,0)
    rp=st.element.get_or_add_rPr()
    rf=rp.rFonts
    if rf is None:rf=OxmlElement('w:rFonts');rp.insert(0,rf)
    rf.set(qn('w:eastAsia'),FONT)
    for c in list(rp):
        if c.tag==qn('w:color'):
            for a in ['themeColor','themeTint','themeShade']:
                c.attrib.pop(qn('w:'+a),None)
    pf=st.paragraph_format
    pf.space_after=Pt(6);pf.line_spacing=1.16;pf.widow_control=True
doc.styles['Normal'].font.size=Pt(11)
doc.styles['Title'].font.size=Pt(21);doc.styles['Title'].font.bold=True
doc.styles['Title'].paragraph_format.space_after=Pt(8)
doc.styles['Subtitle'].font.italic=False;doc.styles['Subtitle'].font.size=Pt(10);doc.styles['Subtitle'].paragraph_format.space_after=Pt(12)
for style,size,before,after in [('Heading 1',15,13,7),('Heading 2',12,9,5)]:
    st=doc.styles[style];st.font.size=Pt(size);st.font.bold=True
    st.paragraph_format.space_before=Pt(before);st.paragraph_format.space_after=Pt(after)
    st.paragraph_format.keep_with_next=True
doc.styles['Caption'].font.size=Pt(9)
doc.styles['Caption'].paragraph_format.line_spacing=1.1
doc.styles['Caption'].paragraph_format.space_before=Pt(4)
doc.styles['Caption'].paragraph_format.space_after=Pt(8)
# Prevent a built-in title rule or themed heading decoration.
for st in doc.styles:
    if st.type==1:
        pp=st.element.find(qn('w:pPr'))
        if pp is not None:
            for x in list(pp):
                if x.tag in (qn('w:pBdr'),qn('w:shd')):pp.remove(x)
p=doc.add_paragraph(style='Title');setfonts(p.add_run(data['title']),21,True,SANS)
p=doc.add_paragraph(style='Subtitle');setfonts(p.add_run(data['subtitle']),10,False,SANS)
hp=sec.header.paragraphs[0];hp.alignment=WD_ALIGN_PARAGRAPH.RIGHT
setfonts(hp.add_run('月球路损地图预测  ·  专家接手说明'),8,False,SANS)
fp=sec.footer.paragraphs[0];fp.alignment=WD_ALIGN_PARAGRAPH.CENTER
setfonts(fp.add_run('第 '),9)
fld=OxmlElement('w:fldSimple');fld.set(qn('w:instr'),'PAGE');fp._p.append(fld)
setfonts(fp.add_run(' 页'),9)
def paragraph(text,style=None,size=11):
    q=doc.add_paragraph(style=style)
    q.paragraph_format.widow_control=True
    setfonts(q.add_run(text),size)
    return q
def hyperlink(par,text,url):
    from docx.opc.constants import RELATIONSHIP_TYPE as RT
    rel=par.part.relate_to(url,RT.HYPERLINK,is_external=True)
    link=OxmlElement('w:hyperlink');link.set(qn('r:id'),rel)
    rr=OxmlElement('w:r');rp=OxmlElement('w:rPr')
    rf=OxmlElement('w:rFonts');rf.set(qn('w:ascii'),'Arial');rf.set(qn('w:hAnsi'),'Arial');rf.set(qn('w:eastAsia'),FONT);rp.append(rf)
    sz=OxmlElement('w:sz');sz.set(qn('w:val'),'18');rp.append(sz)
    color=OxmlElement('w:color');color.set(qn('w:val'),'1F4E79');rp.append(color)
    rr.append(rp);tx=OxmlElement('w:t');tx.text=text;rr.append(tx);link.append(rr);par._p.append(link)
def cell_text(cell,text,head=False,center=False):
    p=cell.paragraphs[0];p.paragraph_format.space_after=Pt(2);p.paragraph_format.space_before=Pt(2);p.paragraph_format.line_spacing=1.1
    p.alignment=WD_ALIGN_PARAGRAPH.CENTER if center else WD_ALIGN_PARAGRAPH.LEFT
    setfonts(p.add_run(str(text)),9.4,head,SANS if head else FONT)
    cell.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
def make_table(b):
    if doc.paragraphs:doc.paragraphs[-1].paragraph_format.keep_with_next=True
    tab=doc.add_table(rows=1,cols=len(b['headers']))
    tab.autofit=False;tab.alignment=WD_TABLE_ALIGNMENT.CENTER
    widths=b['widths'];total=sum(widths)
    if total>17:widths=[v*17/total for v in widths]
    for j,w in enumerate(widths):tab.columns[j].width=Cm(w)
    for c,w in zip(tab.rows[0].cells,widths):c.width=Cm(w)
    props=tab._tbl.tblPr
    borders=OxmlElement('w:tblBorders')
    for edge in ['top','left','bottom','right','insideH','insideV']:
        e=OxmlElement('w:'+edge);e.set(qn('w:val'),'single');e.set(qn('w:sz'),'4');e.set(qn('w:color'),'D9D9D9');borders.append(e)
    props.append(borders)
    mar=OxmlElement('w:tblCellMar')
    for edge,val in [('top','55'),('bottom','55'),('left','90'),('right','90')]:
        e=OxmlElement('w:'+edge);e.set(qn('w:w'),val);e.set(qn('w:type'),'dxa');mar.append(e)
    props.append(mar)
    for j,txt in enumerate(b['headers']):
        c=tab.rows[0].cells[j];cell_text(c,txt,True,True)
        sh=OxmlElement('w:shd');sh.set(qn('w:fill'),'E8EEF3');c._tc.get_or_add_tcPr().append(sh)
    rh=OxmlElement('w:tblHeader');tab.rows[0]._tr.get_or_add_trPr().append(rh)
    for vals in b['rows']:
        row=tab.add_row()
        for j,(c,w,txt) in enumerate(zip(row.cells,widths,vals)):
            c.width=Cm(w)
            numeric=j>0 and len(str(txt))<45 and (any(x.isdigit() for x in str(txt)) or txt in ['—'])
            cell_text(c,txt,False,numeric)
    for row_i,row in enumerate(tab.rows):
        for c in row.cells:
            for p in c.paragraphs:
                if row_i==0 or ((len(b['rows'])<=5 or b.get('keep_together',False)) and row_i<len(tab.rows)-1):p.paragraph_format.keep_with_next=True
        ns=OxmlElement('w:cantSplit');row._tr.get_or_add_trPr().append(ns)
    spacer=doc.add_paragraph();spacer.paragraph_format.space_after=Pt(0);spacer.paragraph_format.space_before=Pt(0)
    spacer.paragraph_format.line_spacing=1
    setfonts(spacer.add_run(''),2)
for b in data['blocks']:
    typ=b['type']
    if typ=='heading':
        q=doc.add_paragraph(style='Heading '+str(b['level']))
        setfonts(q.add_run(b['text']),15 if b['level']==1 else 12,True,SANS)
        if b['text'] in ['1 比赛任务与评价目标','证据来源']:
            q.paragraph_format.page_break_before=True
    elif typ=='paragraph':paragraph(b['text'])
    elif typ=='bullet':
        q=paragraph(b['text'],size=11);q.paragraph_format.left_indent=Cm(.35);q.paragraph_format.first_line_indent=Cm(-.35)
        q.runs[0].text='• '+q.runs[0].text
    elif typ=='caption':paragraph(b['text'],'Caption',9)
    elif typ=='formula':
        q=paragraph(b['text'],size=11);q.alignment=WD_ALIGN_PARAGRAPH.CENTER
        q.paragraph_format.space_before=Pt(4);q.paragraph_format.space_after=Pt(8)
    elif typ=='table':make_table(b)
    elif typ=='image':
        q=doc.add_paragraph();q.alignment=WD_ALIGN_PARAGRAPH.CENTER;q.paragraph_format.keep_with_next=True
        q.add_run().add_picture(str(root/b['path']),width=Cm(b.get('width_cm',16)))
    elif typ=='reference':
        q=paragraph('['+b['id']+'] '+b['text'],size=9.5)
        q.paragraph_format.space_after=Pt(2);q.paragraph_format.keep_with_next=True
        q=doc.add_paragraph()
        q.paragraph_format.space_after=Pt(6)
        if b['url'].startswith('https://'):hyperlink(q,b['url'],b['url'])
        else:
            # Line-break opportunities at separators without changing path content.
            setfonts(q.add_run(b['url']),9)
    else:raise ValueError(typ)
# Set language and remove proofing noise in saved artifact.
for rp in doc.element.xpath('//w:rPr'):
    lang=OxmlElement('w:lang');lang.set(qn('w:val'),'en-US');lang.set(qn('w:eastAsia'),'zh-CN');rp.append(lang)
doc.core_properties.title=data['title']
doc.core_properties.subject='当前方法 结果 关键瓶颈与证据边界'
doc.core_properties.author='CC 项目'
doc.core_properties.keywords='lunar pathloss C82 C130 expert handoff'
output.parent.mkdir(parents=True,exist_ok=True);doc.save(output)
text='\\n'.join(p.text for p in doc.paragraphs)
checks={'output':str(output),'bytes':output.stat().st_size,'sha256':hashlib.sha256(output.read_bytes()).hexdigest(),'paragraphs':len(doc.paragraphs),'tables':len(doc.tables),'inline_shapes':len(doc.inline_shapes),'has_critical_numbers':all(x in text for x in ['1.094 / 1.202','1.230 / 1.384','2.412764'])}
(root/'docx_structure_check.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2)+chr(10))
print(json.dumps(checks,ensure_ascii=False))
