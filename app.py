import streamlit as st
from rdkit import Chem
from rdkit.Chem import Draw, Descriptors, Lipinski, rdMolDescriptors
import pandas as pd

st.set_page_config(page_title="Steatosis Profiler", layout="wide")

# ---------------------------------------------------------------------------
# Overall chemical property domain of the profiler (Table S3B, Supporting Info)
# ---------------------------------------------------------------------------
PROPERTY_DOMAIN = {
    "Hydrogen Bond Acceptors": (0, 18),
    "Hydrogen Bond Donors": (0, 13),
    "Rotatable Bonds Count": (0, 22),
    "Topological Polar Surface Area": (0, 331.94),
    "Molecular Weight": (54.05, 923.49),
    "XLogP": (-6.56, 8.83),
}

def compute_properties(mol):
    return {
        "Hydrogen Bond Acceptors": Lipinski.NumHAcceptors(mol),
        "Hydrogen Bond Donors": Lipinski.NumHDonors(mol),
        "Rotatable Bonds Count": Lipinski.NumRotatableBonds(mol),
        "Topological Polar Surface Area": rdMolDescriptors.CalcTPSA(mol),
        "Molecular Weight": Descriptors.MolWt(mol),
        "XLogP": Descriptors.MolLogP(mol),
    }

# ---------------------------------------------------------------------------
# The 12 refined structural alerts for hepatic steatosis (Table S5, Supporting Info)
# ---------------------------------------------------------------------------
ALERTS = [
    {
        "id": "1", "chemistry": "Aryloxyphenoxypropionate derivatives",
        "smarts": r"[cx2]1[cx2][cx2]([Ax0]*2*****2)[cx2][cx2][cx2]1[Ax0][Ax0][Ax0](=O)[Ax0]",
        "mie": "Activation of PPAR-\u03b3 and PPAR-\u03b1",
        "aops": ["529"], "precision": 1.0,
    },
    {
        "id": "2", "chemistry": "N-substituted triazoles (azole antifungals)",
        "smarts": r"[Cx0]([*][*])n1[cx2,nx2][nX2][nx2,cx2][nx2,cx2]1",
        "mie": "PXR activation",
        "aops": ["60"], "precision": 0.95,
    },
    {
        "id": "3", "chemistry": "Branched alkyl carboxylic acids",
        "smarts": r"[C;x0][C;x0][C;x0][C;x0]([C;x0][C;x0])[C;x0](=O)[O;x0]",
        "mie": "Modulation of PPAR signalling",
        "aops": ["529"], "precision": 1.0,
    },
    {
        "id": "4", "chemistry": "Nucleoside analogues",
        "smarts": r"C([Cx0][Ox0])AC~[#7;x2][#6;x2][#7;x2]",
        "mie": "Inhibition of mitochondrial DNA polymerase",
        "aops": [], "precision": 0.86,
    },
    {
        "id": "5", "chemistry": "Amide-linked aromatic and polar rings (SDHIs)",
        "smarts": r"[cx2,nx2]1[cx2,nx2][cx2,nx2][cx2,nx2][cx2,nx2]1~[#6x0]~[#7x0]~c1ccccc1",
        "mie": "PXR activation",
        "aops": ["60"], "precision": 1.0,
    },
    {
        "id": "6", "chemistry": "Biaryl structures with a carbon linker",
        "smarts": r"[cx2]1[cx2][cx2][cx2][cx2][cx2]1-[C;x0](~[*])-[cx2]1[cx2][cx2][cx2][cx2][cx2]1",
        "mie": "Interaction with ER, AR, PXR and CAR",
        "aops": ["58", "60"], "precision": 1.0,
    },
    {
        "id": "7", "chemistry": "Pyrethroids",
        "smarts": r"[cx2]1[cx2][cx2]([Ax0]*2*****2)[cx2][cx2]([Cx0]([Ox0](C(=O)C3CC3)))[cx2]1",
        "mie": "Modulation of AMPK and FXR\u2013PPAR-\u03b1\u2013CPT1 pathways",
        "aops": ["61", "529"], "precision": 1.0,
    },
    {
        "id": "8a", "chemistry": "Cationic amphiphilic structures",
        "smarts": r"c1cc([Cx0]c)ccc1[Ax0][Ax0][Ax0]N(C)C",
        "mie": "Dysregulation of AR, HNF4\u03b1, RXR, NR2F1 and PPAR-\u03b1",
        "aops": ["529"], "precision": 1.0,
    },
    {
        "id": "8b", "chemistry": "Cationic amphiphilic structures (tricyclic psychotropics)",
        "smarts": r"[ax2]1[ax2][ax2][ax2][*x3]~2~[*]~[*](~[*x3]~[*x3]~[*x2]~[*x3]12)[A]~[A]~[A]~[NX3](C)C",
        "mie": "Multiple mechanisms reported",
        "aops": ["34", "518", "529"], "precision": 1.0,
    },
    {
        "id": "9", "chemistry": "Phenylureas and N-benzoyl-N'-phenylureas",
        "smarts": r"[ax2]1[ax2][ax2][ax2]([N;x0][C;x0](=O)[N;x0]([#6]))[ax2][ax2]1",
        "mie": "PPAR-\u03b3 agonism",
        "aops": ["529"], "precision": 0.57,
    },
    {
        "id": "10", "chemistry": "Oxadiazoles",
        "smarts": r"[cx2]1[cx2][cx2][cx2][cx2][cx2]1[*x2]2~[*x2](~[AX1])~[*x2]~[*x2](~[A]~[A])~[*x2]~2",
        "mie": "Unclear at present",
        "aops": [], "precision": 1.0,
    },
    {
        "id": "11", "chemistry": "Corticosteroids",
        "smarts": (
            r"O=[C!R]([A]1-,=[$([A]~[AD])]2-,=[A](-,=[A]-,=[A]-,=1)-,=[A]1-,=[A]"
            r"(-,=[A]3-,=[A](-,=[A]-,=[$(C~O)]-,=[A]-,=[A]-,=3)-,=[A]-,=[A]-,=1)"
            r"-,=[$([A]~[AD])]-,=[A]-,=2)[A!R]"
        ),
        "mie": "Glucocorticoid receptor (GR) activation",
        "aops": ["318"], "precision": 1.0,
    },
]

for alert in ALERTS:
    alert["pattern"] = Chem.MolFromSmarts(alert["smarts"])

def format_precision(value):
    """Whole-number precisions display without decimals; others show 2 dp."""
    if float(value).is_integer():
        return f"{int(value)}"
    return f"{value:.2f}"

# ---------------------------------------------------------------------------
# Layout: left sidebar navigation
# ---------------------------------------------------------------------------
with st.sidebar:
    st.image("LJMU image.gif", use_column_width=True)
    st.image("risk-hunter-og.png", use_column_width=True)
    logo_col1, logo_col2 = st.columns(2)
    with logo_col1:
        st.image("quantumtox.png", use_column_width=True)
    with logo_col2:
        st.image("ASPIS.png", use_column_width=True)

    st.markdown("---")
    page = st.radio("Navigation", ["Profiler", "About", "Contact", "Acknowledgement"])

# ---------------------------------------------------------------------------
# Profiler
# ---------------------------------------------------------------------------
if page == "Profiler":
    st.title("Steatosis Profiler")
    st.write(
        "Enter a SMILES string to check the chemical property domain and screen for "
        "structural alerts associated with hepatic steatosis."
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

        # -------------------------------------------------------------
        # Chemical property domain check
        # -------------------------------------------------------------
        st.subheader("Chemical Property Domain Check")
        props = compute_properties(mol)
        integer_properties = {"Hydrogen Bond Acceptors", "Hydrogen Bond Donors",
                               "Rotatable Bonds Count"}
        overall_within = True
        domain_rows = []
        for prop, (lo, hi) in PROPERTY_DOMAIN.items():
            val = props[prop]
            within = lo <= val <= hi
            if not within:
                overall_within = False
            if prop in integer_properties:
                value_str = f"{val:.0f}"
                lo_str = f"{lo:.0f}"
                hi_str = f"{hi:.0f}"
            else:
                value_str = f"{val:.2f}"
                lo_str = f"{lo:.2f}"
                hi_str = f"{hi:.2f}"
            domain_rows.append({
                "Property": prop,
                "Value": value_str,
                "Domain Min": lo_str,
                "Domain Max": hi_str,
                "Within Domain": "Yes" if within else "No",
            })
        st.dataframe(domain_rows)
        if overall_within:
            st.success("Within property domain: Yes")
        else:
            st.error("Within property domain: No")

        # -------------------------------------------------------------
        # Structural alert screening
        # -------------------------------------------------------------
        st.subheader("Structural Alert Screening")
        matches = [a for a in ALERTS if a["pattern"] is not None
                   and mol.HasSubstructMatch(a["pattern"])]

        if matches:
            results_table = pd.DataFrame([{
                "Chemistry": a["chemistry"],
                "SMARTS": a["smarts"],
                "MIE": a["mie"],
                "AOP/s": ", ".join(a["aops"]) if a["aops"] else "-",
                "Precision": format_precision(a['precision']),
            } for a in matches])
            st.dataframe(results_table, use_container_width=True)
            st.warning(
                "Results indicate that this chemical may be steatogenic. "
                "Further investigation recommended."
            )
        else:
            st.success(
                "Results indicate that this chemical may not be steatogenic. "
                "However, it is recommended that additional evidence is sought to "
                "inform decision-making."
            )

# ---------------------------------------------------------------------------
# About
# ---------------------------------------------------------------------------
elif page == "About":
    st.header("About Steatosis Profiler")
    st.markdown(
        """
        ### Purpose
        Steatosis Profiler checks whether a query chemical falls within the chemical
        property domain of the underlying dataset, and screens it against 12 refined
        structural alerts, encoded as SMARTS patterns, indicative of hepatic steatosis.

        Each alert is linked to a putative Molecular Initiating Event (MIE) and, where
        established, the relevant Adverse Outcome Pathway(s) (AOPs).
        """
    )

    st.markdown("### Development")
    st.image("workflow_schematic.png", use_column_width=True)
    st.markdown(
        """
        This profiler was developed using rules derived from Molecular Initiating Events
        (MIEs) linked to steatosis and fragments generated using SARpy, which were then
        refined using expert judgement.

        The manuscript describing the profiler is currently under revision. Associated
        (meta)data can be found on Zenodo:
        [10.5281/zenodo.20236679](https://doi.org/10.5281/zenodo.20236679).
        """
    )

    st.markdown("### Overview of Alerts within Profiler")
    overview_table = pd.DataFrame([{
        "Alert ID": a["id"],
        "Chemistry": a["chemistry"],
        "Putative Mechanism of Action": a["mie"],
        "Relevant AOP/s": ", ".join(a["aops"]) if a["aops"] else "-",
        "Precision": format_precision(a['precision']),
    } for a in ALERTS])
    st.dataframe(overview_table, use_container_width=True)

elif page == "Contact":
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

elif page == "Acknowledgement":
    st.title("Acknowledgement")
    st.write(
        """
        This work was supported by the project RISK-HUNT3R: RISK assessment of chemicals
        integrating human centric Next generation Testing strategies promoting the 3Rs.
        RISK-HUNT3R has received funding from the European Union's Horizon 2020 research
        and innovation programme under grant agreement No 964537 and is part of the ASPIS
        cluster; and QUANTUM-TOX \u2013 Revolutionizing Computational Toxicology with
        Electronic Structure Descriptors and Artificial Intelligence (QUANTUM-TOX)
        HORIZON-EIC-2023-PATHFINDEROPEN-01 Project number: 101130724.
        """
    )
