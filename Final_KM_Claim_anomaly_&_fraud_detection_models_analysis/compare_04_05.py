"""
export_comparison_to_excel.py
=============================
Reads the REAL step-4 and step-5 claimant/dealer outputs and writes an
interactive Excel workbook (same template as the illustrative one) populated
with your true numbers.

Live Excel formulas mean the audience can still edit ratings and watch the
agreement %, the matrix, and Cohen's kappa recompute.

INPUT (auto-detected in ./data):
  04_claims_riskscored_by_claimant.parquet
  05_claims_riskscored_4models_by_claimant.parquet
  (and the _by_dealer versions, optional)

OUTPUT:
  data/KM_Step4_vs_Step5_Comparison_REAL.xlsx

USAGE:
  python export_comparison_to_excel.py
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.datavalidation import DataValidation

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"

BLUE="1F4E78"; LTBLUE="DDEBF7"; AMBER="FFF2CC"
hdr=Font(bold=True,color="FFFFFF",name="Arial",size=10)
bold=Font(bold=True,name="Arial",size=10)
normal=Font(name="Arial",size=10)
center=Alignment(horizontal="center",vertical="center")
left=Alignment(horizontal="left",vertical="center")
fill_hdr=PatternFill("solid",fgColor=BLUE)
thin=Side(style="thin",color="BBBBBB"); border=Border(left=thin,right=thin,top=thin,bottom=thin)


def find(*cands):
    for c in cands:
        p = DATA / c
        if p.exists():
            return p
    return None


def load_pair(entity):
    """Return a merged dataframe with columns: entity, step4_level, step5_level,
    and any evidence columns present in the step-4 file."""
    prefix = "claimant" if entity == "ClaimantName" else "dealer"
    f4 = find(f"04_claims_riskscored_by_{prefix}.parquet")
    f5 = find(f"05_claims_riskscored_4models_by_{prefix}.parquet")
    if f4 is None or f5 is None:
        return None
    d4 = pd.read_parquet(f4)
    d5 = pd.read_parquet(f5)
    col = f"{prefix}_risk_level"
    if col not in d4.columns or col not in d5.columns:
        raise ValueError(f"'{col}' not found in the {prefix} files.")

    evidence = [c for c in ["claim_count", "high_count", "raw_high_rate", "wilson_lower"]
                if c in d4.columns]
    left_cols = [entity, col] + evidence
    m = d4[left_cols].merge(d5[[entity, col]], on=entity, suffixes=("_step4", "_step5"))
    m = m.rename(columns={f"{col}_step4": "step4_level", f"{col}_step5": "step5_level"})
    m["step4_level"] = m["step4_level"].astype(str).str.title()
    m["step5_level"] = m["step5_level"].astype(str).str.title()
    return m, entity, evidence


def add_ratings_sheet(wb, m, entity, evidence, title):
    ws = wb.create_sheet(title)
    ws.cell(1, 1, f"{title} - Step 4 vs Step 5 (editable)").font = Font(bold=True, size=13, color=BLUE, name="Arial")
    ws.cell(2, 1, "Change the Step 4 / Step 5 columns; the Summary sheet updates live.").font = Font(italic=True, size=9, name="Arial", color="666666")

    cols = [entity] + evidence + ["step4_level", "step5_level", "agree?"]
    r0 = 4
    for j, h in enumerate(cols, 1):
        c = ws.cell(r0, j, h); c.font = hdr; c.fill = fill_hdr; c.alignment = center; c.border = border

    dv = DataValidation(type="list", formula1='"High,Medium,Low"', allow_blank=False)
    ws.add_data_validation(dv)

    s4_idx = 1 + len(evidence) + 1   # column number of step4_level
    s5_idx = s4_idx + 1
    agree_idx = s5_idx + 1

    for i, row in enumerate(m.itertuples(index=False), start=r0 + 1):
        vals = [getattr(row, entity)] + [getattr(row, e) for e in evidence] + [row.step4_level, row.step5_level]
        for j, v in enumerate(vals, 1):
            c = ws.cell(i, j, v); c.border = border; c.alignment = center if j > 1 else left; c.font = normal
            if j == 1: c.font = bold
            if j in (s4_idx, s5_idx): c.fill = PatternFill("solid", fgColor=LTBLUE)
        dv.add(ws.cell(i, s4_idx)); dv.add(ws.cell(i, s5_idx))
        L4 = get_column_letter(s4_idx); L5 = get_column_letter(s5_idx)
        ws.cell(i, agree_idx).value = f'=IF({L4}{i}={L5}{i},"YES","NO")'

    last = r0 + len(m)
    widths = [22] + [12]*len(evidence) + [12, 12, 9]
    for j, w in enumerate(widths, 1):
        ws.column_dimensions[get_column_letter(j)].width = w
    ws.freeze_panes = f"A{r0+1}"
    return ws.title, r0, last, s4_idx, s5_idx


def add_summary_sheet(wb, ratings_title, r0, last, s4_idx, s5_idx):
    s = wb.create_sheet(f"Summary ({ratings_title})")
    L4 = get_column_letter(s4_idx); L5 = get_column_letter(s5_idx)
    rng4 = f"'{ratings_title}'!${L4}${r0+1}:${L4}${last}"
    rng5 = f"'{ratings_title}'!${L5}${r0+1}:${L5}${last}"

    s["A1"] = f"Comparison Summary - {ratings_title} (recalculates live)"
    s["A1"].font = Font(bold=True, size=13, color=BLUE, name="Arial")
    s["A3"]="Total"; s["B3"]=f"=COUNTA({rng4})"
    s["A4"]="Exact agreement (count)"; s["B4"]=f'=SUMPRODUCT(--({rng4}={rng5}))'
    s["A5"]="Exact agreement (%)"; s["B5"]="=B4/B3"; s["B5"].number_format="0.0%"
    for rr in (3,4,5): s[f"A{rr}"].font=bold

    labels=["High","Medium","Low"]
    s["A7"]="Agreement matrix (rows = Step 4, cols = Step 5)"; s["A7"].font=bold
    for j,lab in enumerate(labels):
        c=s.cell(8,2+j,lab); c.font=hdr; c.fill=fill_hdr; c.alignment=center
    for i,lab in enumerate(labels):
        rc=s.cell(9+i,1,lab); rc.font=hdr; rc.fill=fill_hdr; rc.alignment=center
        for j,lab2 in enumerate(labels):
            s.cell(9+i,2+j).value=f'=SUMPRODUCT(--({rng4}="{lab}"),--({rng5}="{lab2}"))'
            s.cell(9+i,2+j).alignment=center; s.cell(9+i,2+j).border=border
    for i in range(3): s.cell(9+i,5).value=f'=SUM(B{9+i}:D{9+i})'
    for j in range(3): s.cell(12,2+j).value=f'=SUM({get_column_letter(2+j)}9:{get_column_letter(2+j)}11)'

    s["A14"]="Cohen's kappa (agreement beyond chance)"; s["A14"].font=bold
    s["A15"]="po (observed)"; s["B15"]="=B5"
    s["A16"]="pe (expected)"; s["B16"]="=(E9*B12+E10*C12+E11*D12)/(B3*B3)"
    s["A17"]="kappa"; s["B17"]="=(B15-B16)/(1-B16)"
    s["B17"].font=Font(bold=True,size=12,color="C00000",name="Arial")
    s["A18"]="Interpretation"; s["A18"].font=bold
    s["B18"]='=IF(B17<0,"worse than chance",IF(B17<=0.2,"slight",IF(B17<=0.4,"fair",IF(B17<=0.6,"moderate",IF(B17<=0.8,"substantial","almost perfect")))))'
    for rr in (15,16,17): s[f"A{rr}"].font=bold
    for col,w in [("A",34),("B",16),("C",12),("D",12),("E",12)]: s.column_dimensions[col].width=w


def add_disagreements_sheet(wb, m, entity, evidence, title):
    d = wb.create_sheet(f"Disagree ({title})")
    d["A1"]=f"Disagreements - {title} (review priority)"; d["A1"].font=Font(bold=True,size=13,color="BF9000",name="Arial")
    cols=[entity]+evidence+["step4_level","step5_level"]
    for j,h in enumerate(cols,1):
        c=d.cell(3,j,h); c.font=hdr; c.fill=fill_hdr; c.alignment=center; c.border=border
    dis=m[m["step4_level"]!=m["step5_level"]]
    for i,row in enumerate(dis.itertuples(index=False),start=4):
        vals=[getattr(row,entity)]+[getattr(row,e) for e in evidence]+[row.step4_level,row.step5_level]
        for j,v in enumerate(vals,1):
            c=d.cell(i,j,v); c.border=border; c.alignment=center if j>1 else left; c.font=(bold if j==1 else normal)
    if len(dis): d.auto_filter.ref=f"A3:{get_column_letter(len(cols))}{3+len(dis)}"
    widths=[22]+[12]*len(evidence)+[12,12]
    for j,w in enumerate(widths,1): d.column_dimensions[get_column_letter(j)].width=w
    d.freeze_panes="A4"
    return len(dis)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--out", default=str(DATA/"KM_Step4_vs_Step5_Comparison_REAL.xlsx"))
    args=ap.parse_args()

    wb=Workbook(); wb.remove(wb.active)
    built=[]
    for entity, nice in [("ClaimantName","Claimants"), ("DealerName","Dealers")]:
        pair=load_pair(entity)
        if pair is None:
            print(f"[skip] {nice}: files not found")
            continue
        m, ent, evidence = pair
        title=nice
        rtitle,r0,last,s4,s5=add_ratings_sheet(wb,m,ent,evidence,title)
        add_summary_sheet(wb,rtitle,r0,last,s4,s5)
        nd=add_disagreements_sheet(wb,m,ent,evidence,title)
        built.append((nice,len(m),nd))
        print(f"[ok] {nice}: {len(m)} entities, {nd} disagreements")

    if not built:
        print("No input files found in ./data. Run steps 4 and 5 first.")
        return

    # read-me sheet first
    rm=wb.create_sheet("Read me",0)
    rm["A1"]="KM Claims - Step 4 vs Step 5 (REAL data, interactive)"; rm["A1"].font=Font(bold=True,size=14,color=BLUE,name="Arial")
    lines=["","Populated from your real step-4 and step-5 output files.","",
           "Sheets per entity: Ratings (editable) | Summary (live kappa) | Disagree (review list).","",
           "Edit a rating in a Ratings sheet and the matching Summary sheet updates live.",""]
    for n,cnt,nd in built:
        lines.append(f"  {n}: {cnt} entities, {nd} disagreements to review")
    for i,t in enumerate(lines,3):
        rm.cell(i,1,t).font=Font(name="Arial",size=10)
    rm.column_dimensions["A"].width=95

    out=Path(args.out)
    wb.save(out)
    print(f"[save] {out}")


if __name__ == "__main__":
    main()
