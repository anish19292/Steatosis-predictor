import streamlit as st
from rdkit import Chem
from rdkit.Chem import Draw
import pandas as pd

st.set_page_config(page_title="Steatosis Structural Alert Profiler", layout="wide")

# ---------------------------------------------------------------------------
# AOP-Wiki reference titles for the AOP IDs cited against each alert
# ---------------------------------------------------------------------------
AOP_TITLES = {
    "34": "LXR activation leading to hepatic steatosis",
    "58": "NR1I3 (CAR) suppression leading to hepatic steatosis",
    "60": "NR1I2 (PXR) activation leading to hepatic steatosis",
    "61": "NFE2L2/FXR activation leading to hepatic steatosis",
    "518": "LXR activation leads to liver steatosis",
    "529": "Xenobiotic binding to PPARs causes dysregulation of lipid "
           "metabolism leading to liver steatosis",
}

def aop_link(aop_id):
    return f"https://aopwiki.org/aops/{aop_id}"

# ---------------------------------------------------------------------------
# The 12 refined structural alerts for hepatic steatosis
# (Gomatam et al., Refinement of Structural Alerts for Hepatic Steatosis)
# ---------------------------------------------------------------------------
ALERTS = [
    {
        "id": "1",
        "chemistry": "Aryloxyphenoxypropionate derivatives",
        "smarts": r"[cx2]1[cx2][cx2]([Ax0]*2*****2)[cx2][cx2][cx2]1[Ax0][Ax0][Ax0](=O)[Ax0]",
        "mie": "Activation of PPAR-\u03b3 and PPAR-\u03b1",
        "aops": ["529"],
        "description": (
            "Aryloxyphenoxypropionates ('fop' herbicides) are composed of oxygen-linked "
            "aryl/heteroaryl rings. Although their herbicidal action targets acetyl-CoA "
            "carboxylase in plant plastids, they also increase expression of PPAR-\u03b1 and "
            "PPAR-\u03b3 and promote lipid accumulation in mammalian cell and rodent models."
        ),
        "tp": 4, "fp": 0, "precision": 1.0,
        "examples": ["Diclofop Methyl", "Fluazifop", "Propaquizafop", "Quizalofop-P-tefuryl"],
    },
    {
        "id": "2",
        "chemistry": "N-substituted triazoles (azole antifungals)",
        "smarts": r"[Cx0]([*][*])n1[cx2,nx2][nX2][nx2,cx2][nx2,cx2]1",
        "mie": "PXR activation",
        "aops": ["60"],
        "description": (
            "Defines an alkyl substituent on a five-membered aromatic nitrogen-containing "
            "ring typical of imidazole, pyrazole and triazole systems. Azole antifungals "
            "such as propiconazole and tebuconazole are mechanistically linked to PXR "
            "activation in human liver cells, potentially leading to triglyceride "
            "accumulation."
        ),
        "tp": 19, "fp": 1, "precision": 0.95,
        "examples": ["Bitertanol", "Bromuconazole", "Cyproconazole", "Difenoconazole",
                     "Epoxiconazole", "Flusilazole", "Flutriafol", "Hexaconazole",
                     "Imazalil", "Ipconazole", "Metconazole", "Paclobutrazol",
                     "Tebuconazole", "Tetraconazole", "Triadimefon", "Triadimenol",
                     "Triflumizole", "Propiconazole", "Myclobutanil"],
    },
    {
        "id": "3",
        "chemistry": "Branched alkyl carboxylic acids",
        "smarts": r"[C;x0][C;x0][C;x0][C;x0]([C;x0][C;x0])[C;x0](=O)[O;x0]",
        "mie": "Modulation of PPAR signalling",
        "aops": ["529"],
        "description": (
            "Captures branched aliphatic carboxylic acid analogues of varying chain length, "
            "typified by the antiepileptic valproic acid. Proposed mechanisms include Nrf2- "
            "and PPAR-\u03b3-mediated triglyceride accumulation and CD36-dependent lipid uptake "
            "preceding lipid droplet formation."
        ),
        "tp": 6, "fp": 0, "precision": 1.0,
        "examples": ["Valproic acid", "2-Propyl-heptanoic acid", "2-Ethyl-heptanoic acid",
                     "2-Propyl-hexanoic acid", "2-Ethyl-hexanoic acid",
                     "2-Propyl-4-pentenoic acid"],
    },
    {
        "id": "4",
        "chemistry": "Nucleoside analogues",
        "smarts": r"C([Cx0][Ox0])AC~[#7;x2][#6;x2][#7;x2]",
        "mie": "Inhibition of mitochondrial DNA polymerase",
        "aops": [],
        "description": (
            "Describes a nucleoside-like motif \u2013 a furanose/deoxyribose sugar ring "
            "attached to an aromatic nitrogen \u2013 representative of antiviral nucleoside "
            "analogues. Hepatotoxicity is well documented (e.g. fialuridine), thought to "
            "arise via mitochondrial injury from inhibition of mitochondrial DNA "
            "polymerase, leading to micro- and macro-vesicular steatosis and lactic "
            "acidosis."
        ),
        "tp": 4, "fp": 1, "precision": 0.80,
        "examples": ["Didanosine", "Fialuridine", "Stavudine",
                     "3'-Azido-3'-deoxythymidine"],
    },
    {
        "id": "5",
        "chemistry": "Amide-linked aromatic and polar rings (SDHIs)",
        "smarts": r"[cx2,nx2]1[cx2,nx2][cx2,nx2][cx2,nx2][cx2,nx2]1~[#6x0]~[#7x0]~c1ccccc1",
        "mie": "PXR activation",
        "aops": ["60"],
        "description": (
            "Captures succinate dehydrogenase inhibitor (SDHI) fungicides, typically "
            "composed of a pyrazole ring and a further aromatic moiety joined by an amide "
            "linkage. Strong evidence links SDHIs to PXR activation as the likely "
            "mechanism underlying steatosis induction."
        ),
        "tp": 4, "fp": 0, "precision": 1.0,
        "examples": ["Bixafen", "Isopyrazam", "Metazachlor", "Triflumizole",
                     "Fluxapyroxad"],
    },
    {
        "id": "6",
        "chemistry": "Biaryl structures with a carbon linker",
        "smarts": r"[cx2]1[cx2][cx2][cx2][cx2][cx2]1-[C;x0](~[*])-[cx2]1[cx2][cx2][cx2][cx2][cx2]1",
        "mie": "Interaction with ER, AR, PXR and CAR",
        "aops": ["58", "60"],
        "description": (
            "Two benzene rings linked by a single carbon. Captures the organochlorine "
            "insecticides DDT and DDE, the anticancer drug tamoxifen, and several "
            "fungicides. Proposed mechanisms include dysregulation of genes involved in "
            "fatty acid uptake (CD36), lipid metabolism (ALB, AKT1) and fatty acid "
            "biosynthesis (SCD2, TP53). The alert does not define a single chemical class "
            "beyond the biaryl motif, so caution is warranted for grouping/read-across use."
        ),
        "tp": 7, "fp": 0, "precision": 1.0,
        "examples": ["Tamoxifen", "DDT", "Dimethomorph", "Fenarimol", "Flutriafol",
                     "Metrafenone", "p,p'-DDE"],
    },
    {
        "id": "7",
        "chemistry": "Pyrethroids",
        "smarts": r"[cx2]1[cx2][cx2]([Ax0]*2*****2)[cx2][cx2]([Cx0]([Ox0](C(=O)C3CC3)))[cx2]1",
        "mie": "Modulation of AMPK and FXR\u2013PPAR-\u03b1\u2013CPT1 pathways",
        "aops": ["61", "529"],
        "description": (
            "Synthetic cyclopropanecarboxylic acid ester analogues of naturally occurring "
            "pyrethrins. Associated with increased triglyceride accumulation, fatty acid "
            "synthesis and dysregulated lipid metabolism in HepG2 cells, via AMPK-dependent "
            "lipogenesis and reduced fatty acid oxidation through the FXR\u2013PPAR-\u03b1\u2013CPT1 "
            "pathway."
        ),
        "tp": 3, "fp": 0, "precision": 1.0,
        "examples": ["alpha-Cypermethrin", "Deltamethrin", "Permethrin"],
    },
    {
        "id": "8a",
        "chemistry": "Cationic amphiphilic structures",
        "smarts": r"c1cc([Cx0]c)ccc1[Ax0][Ax0][Ax0]N(C)C",
        "mie": "Dysregulation of AR, HNF4\u03b1, RXR, NR2F1 and PPAR-\u03b1",
        "aops": ["529"],
        "description": (
            "Cationic amphiphilic drugs (CADs) combine a lipophilic ring system (LogP > 1) "
            "with a secondary/tertiary amine (pKa > 6.5). These features promote ionisation "
            "in the acidic lysosomal environment, impairing diffusion across intracellular "
            "membranes and leading to hepatic accumulation. CADs are strongly associated "
            "with steatohepatitis, an advanced form of steatosis combining lipid build-up "
            "with inflammation."
        ),
        "tp": 2, "fp": 0, "precision": 1.0,
        "examples": ["Tamoxifen", "Amiodarone"],
    },
    {
        "id": "8b",
        "chemistry": "Cationic amphiphilic structures (tricyclic psychotropics)",
        "smarts": r"[ax2]1[ax2][ax2][ax2][*x3]~2~[*]~[*](~[*x3]~[*x3]~[*x2]~[*x3]12)[A]~[A]~[A]~[NX3](C)C",
        "mie": "Multiple mechanisms reported",
        "aops": ["34", "518", "529"],
        "description": (
            "A second, tricyclic cationic amphiphilic pattern, capturing atypical "
            "antipsychotics with a fused tricyclic core and a terminal tertiary amine "
            "side-chain. Several mechanisms have been proposed for steatosis induction "
            "with this class, including LXR and PPAR pathway involvement."
        ),
        "tp": 2, "fp": 0, "precision": 1.0,
        "examples": ["Clozapine", "Olanzapine"],
    },
    {
        "id": "9",
        "chemistry": "Phenylureas and N-benzoyl-N'-phenylureas",
        "smarts": r"[ax2]1[ax2][ax2][ax2]([N;x0][C;x0](=O)[N;x0]([#6]))[ax2][ax2]1",
        "mie": "PPAR-\u03b3 agonism",
        "aops": ["529"],
        "description": (
            "Captures the phenylurea subunit common to phenylureas and benzoylphenylureas "
            "(BPUs), widely used as insect growth regulators. BPUs such as flufenoxuron "
            "have been shown to potently agonise PPAR-\u03b3 in HepG2 cells, decreasing "
            "expression of TCA-cycle enzymes and increasing availability of glycerol "
            "precursors for triglyceride accumulation."
        ),
        "tp": 4, "fp": 3, "precision": 0.57,
        "examples": ["Flufenoxuron", "Isoproturon", "Teflubenzuron"],
    },
    {
        "id": "10",
        "chemistry": "Oxadiazoles",
        "smarts": r"[cx2]1[cx2][cx2][cx2][cx2][cx2]1[*x2]2~[*x2](~[AX1])~[*x2]~[*x2](~[A]~[A])~[*x2]~2",
        "mie": "Unclear at present",
        "aops": [],
        "description": (
            "A six-membered aromatic ring linked, by a single bond, to a five-membered "
            "ring. Captures the herbicides oxadiazon and oxadiargyl, which act through "
            "inhibition of protoporphyrinogen oxidase. Oxadiargyl is classified by EFSA "
            "in Steatosis CAG2b (propensity to cause hepatocellular fatty change in vivo), "
            "but the mechanism linking this chemistry to steatosis remains unclear."
        ),
        "tp": 1, "fp": 1, "precision": 0.50,
        "examples": ["Oxadiazon", "Oxadiargyl"],
    },
    {
        "id": "11",
        "chemistry": "Corticosteroids",
        "smarts": (
            r"O=[C!R]([A]1-,=[$([A]~[AD])]2-,=[A](-,=[A]-,=[A]-,=1)-,=[A]1-,=[A]"
            r"(-,=[A]3-,=[A](-,=[A]-,=[$(C~O)]-,=[A]-,=[A]-,=3)-,=[A]-,=[A]-,=1)"
            r"-,=[$([A]~[AD])]-,=[A]-,=2)[A!R]"
        ),
        "mie": "Glucocorticoid receptor (GR) activation",
        "aops": [],
        "description": (
            "Captures the steroid nucleus common to corticosteroid drugs. GR activation is "
            "well established as promoting hepatic lipid accumulation, consistent with "
            "steatosis reported for corticosteroid exposure in the clinical literature."
        ),
        "tp": 7, "fp": 0, "precision": 1.0,
        "examples": ["Dexamethasone", "Betamethasone", "Cortisone", "Hydrocortisone",
                     "Prednisolone", "Methylprednisolone", "Prednisone"],
    },
]

for alert in ALERTS:
    alert["pattern"] = Chem.MolFromSmarts(alert["smarts"])

# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
spacer1, col1, spacer2, col2, spacer3 = st.columns([0.05, 0.4, 0.2, 0.4, 0.05])
with col1:
    st.image("LJMU image.gif", use_column_width=True)
with col2:
    st.image("risk-hunter-og.png", use_column_width=True)

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Predictor", "Alert Details", "About", "Contact", "Acknowledgement"]
)

# ---------------------------------------------------------------------------
# Tab 1: Predictor
# ---------------------------------------------------------------------------
with tab1:
    st.title("Steatosis Structural Alert Profiler")
    st.write(
        "Enter a SMILES string to screen for structural alerts associated with hepatic "
        "steatosis. See the **Alert Details** tab for the full description, Molecular "
        "Initiating Event (MIE) and relevant Adverse Outcome Pathway(s) (AOPs) for each alert."
    )

    smiles_input = st.text_input("Enter SMILES:", "CCCC(CCC)C(O)=O")

    mol = None
    if smiles_input:
        try:
            mol = Chem.MolFromSmiles(smiles_input)
            if mol is None:
                st.error("Invalid SMILES input.")
        except Exception as e:
            st.error(f"Error processing SMILES: {e}")

    if mol:
        st.subheader("Molecule Structure")
        st.image(Draw.MolToImage(mol, size=(300, 300)))

        matches = [a for a in ALERTS if a["pattern"] is not None
                   and mol.HasSubstructMatch(a["pattern"])]

        if matches:
            st.subheader(f"{len(matches)} Matching Structural Alert(s)")
            results_df = pd.DataFrame([{
                "Alert ID": a["id"],
                "Chemistry": a["chemistry"],
                "MIE": a["mie"],
                "Relevant AOP(s)": ", ".join(a["aops"]) if a["aops"] else "-",
                "Precision": a["precision"],
            } for a in matches])
            st.dataframe(results_df, use_container_width=True, hide_index=True)
        else:
            st.info("No matching structural alerts found for the given molecule.")

# ---------------------------------------------------------------------------
# Tab 2: Alert Details
# ---------------------------------------------------------------------------
with tab2:
    st.title("Structural Alert Details")
    st.write(
        "Full reference for the 12 structural alerts used by the Predictor, including "
        "each alert's putative Molecular Initiating Event (MIE), relevant Adverse Outcome "
        "Pathway(s) (AOPs), and SMARTS pattern."
    )

    for alert in ALERTS:
        aop_lines = []
        for aop_id in alert["aops"]:
            title = AOP_TITLES.get(aop_id, "")
            aop_lines.append(f"- **AOP {aop_id}** \u2013 {title} "
                              f"([aopwiki.org]({aop_link(aop_id)}))")
        aop_text = "\n".join(aop_lines) if aop_lines else "_No AOP currently assigned in AOP-Wiki._"

        with st.expander(f"Alert {alert['id']}: {alert['chemistry']}"):
            st.markdown(f"**Molecular Initiating Event / MoA:** {alert['mie']}")
            st.markdown("**Relevant AOP(s):**")
            st.markdown(aop_text)
            st.markdown(f"**Description:** {alert['description']}")
            st.markdown(
                f"**Performance in curated dataset:** {alert['tp']} true positives, "
                f"{alert['fp']} false positives (precision = {alert['precision']:.2f})"
            )
            st.code(alert["smarts"], language="text")

# ---------------------------------------------------------------------------
# Tab 3: About
# ---------------------------------------------------------------------------
with tab3:
    st.header("About the Steatosis Structural Alert Profiler")
    st.markdown(
        """
        ### Purpose
        This profiler queries a set of 12 refined structural alerts, encoded as SMARTS
        patterns, indicative of hepatic steatosis. Each alert is grounded in a putative
        Molecular Initiating Event (MIE) and, where an established Adverse Outcome Pathway
        (AOP) exists in the AOP-Wiki, this is cited alongside the alert.

        The alerts were developed through refinement of an initial set of 214 structural
        rules (Mellor et al., 2016), combined with fragment generation against a curated
        dataset of 1,378 substances judged for steatosis potential. Refinement criteria
        included a minimum precision of 0.6 and exclusion of overly generic substructures.

        ### Molecular Initiating Events (MIEs) and Adverse Outcome Pathways (AOPs)
        As critical regulators of lipid accumulation and metabolism, nuclear receptors play
        a key role in the onset and progression of steatosis. Several AOPs link modulation
        of these receptors to hepatic lipid and triglyceride accumulation, including AOPs
        for PPAR (e.g. AOP 529), PXR (AOP 60), CAR (AOP 58), FXR (AOP 61) and LXR
        (AOPs 34/518).

        #### Key Nuclear Receptors:

        | Nuclear Receptor Name                          | Abbreviation | Nomenclature Identification |
        |-----------------------------------------------|--------------|------------------------------|
        | Aryl hydrocarbon receptor                      | AHR          | bHLHe76                      |
        | Constitutive androstane receptor               | CAR          | NR1I3                        |
        | Estrogen receptor                              | ER           | NR3A1/2                      |
        | Farnesoid X receptor                           | FXR          | NR1H4/5                      |
        | Glucocorticoid receptor                        | GR           | NR3C1                        |
        | Liver X receptor                               | LXR          | NR1H2/3                      |
        | Peroxisome proliferator-activated receptor     | PPAR         | NR1C1-3                      |
        | Pregnane X receptor                            | PXR          | NR1I2                        |
        | Retinoic acid receptor                         | RAR          | NR1B1-3                      |
        | Retinoid X receptor                            | RXR          | \u2014                            |

        ### Intended Use
        This profiler is intended for use within tiered, weight-of-evidence frameworks that
        integrate data from multiple New Approach Methodologies (NAMs), rather than as a
        standalone tool for direct hazard assessment. It may also assist in identifying
        analogues for grouping and read-across.

        More information on the original alert set can be found here:
        [10.1021/acs.chemrestox.5b00480](https://doi.org/10.1021/acs.chemrestox.5b00480)
        """
    )

with tab4:
    st.header("About Us")
    st.markdown(
        """
        <div style="font-size:16px; line-height:1.8">

        <p><strong>Anish Gomatam</strong> \u2013 Postdoctoral Researcher<br>
        Email: <a href="mailto:A.Gomatam@ljmu.ac.uk">A.Gomatam@ljmu.ac.uk</a></p>

        <p><strong>James Firman</strong> \u2013 Postdoctoral Researcher<br>
        Email: <a href="mailto:J.W.Firman@ljmu.ac.uk">J.W.Firman@ljmu.ac.uk</a></p>

        <p><strong>Georgios Chrysochoou</strong> \u2013 Postdoctoral Researcher<br>
        Email: <a href="mailto:G.Chrysochoou@ljmu.ac.uk">G.Chrysochoou@ljmu.ac.uk</a></p>

        <p><strong>Prof. Mark Cronin</strong> \u2013 Principal Investigator<br>
        Email: <a href="mailto:M.T.Cronin@ljmu.ac.uk">M.T.Cronin@ljmu.ac.uk</a></p>

        <p><strong>Affiliation:</strong><br>
        School of Pharmacy and Biomolecular Sciences,<br>
        Liverpool John Moores University,<br>
        Byrom Street, Liverpool L3 3AF, United Kingdom</p>

        </div>
        """,
        unsafe_allow_html=True
    )

with tab5:
    st.title("Acknowledgement")
    st.write(
        """
        This project receives funding from the European Union's Horizon 2020 Research and
        Innovation programme under Grant Agreement No. 964537 (RISK-HUNT3R),
        and it is part of the ASPIS cluster.
        """
    )
