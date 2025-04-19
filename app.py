import streamlit as st
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import Descriptors
from rdkit.Chem import Lipinski

# Define SMARTS patterns and their associated MIEs with chemical property domains
smarts_mie_mapping = {
    "C(=C\\c1ccccc1)\c1ccccc1": {
        "AhR": {"Domain": {"HBD": (0, 6), "MW": (180, 900), "HBA": (0, 10), "XLogP": (None, 8)}}
    },
    "c1nc2ccccc2s1": {
        "AhR": {"Domain": {"HBD": (0, 6), "MW": (180, 900), "HBA": (0, 10), "XLogP": (None, 8)}}
    },
    "c1c*o*1": {
        "AhR": {"Domain": {"HBD": (0, 10), "MW": (140, 700), "HBA": (0, 15), "XLogP": (-2, None)}},
        "ER": {"Domain": {"HBD": (0, 10), "MW": (140, 700), "HBA": (0, 15), "XLogP": (-2, None)}}
    },
    "[#7,#6,#8,#16]1[#7,#6,#8,#16][#7,#6,#8,#16][#7,#6,#8,#16]([#7,#6,#8,#16]1)-c1ccccc1": {
        "AhR": {"Domain": {"HBD": (0, 15), "MW": (300, 610), "HBA": (0, 10), "XLogP": (0, None)}},
        "ER": {"Domain": {"HBD": (0, 10), "MW": (140, 700), "HBA": (0, 15), "XLogP": (-2, None)}},
        "GR": {"Domain": {"HBD": (0, 15), "MW": (300, 610), "HBA": (0, 10), "XLogP": (0, None)}},
        "PXR": {"Domain": {"HBD": (0, 15), "MW": (300, 610), "HBA": (0, 10), "XLogP": (0, None)}},
    },
    "[#8,#7,#6]~1~[#8,#7,#6]~[#8,#7,#6]~c2ccccc2~[#8,#7,#6]~1": {
        "AhR": {"Domain": {"HBD": (0, 6), "MW": (180, 900), "HBA": (0, 10), "XLogP": (None, 8)}}
    },
    "[#6]-[#7]-c1ccccc1-[#9,#17]": {
        "AhR": {"Domain": {"HBD": (0, 6), "MW": (180, 900), "HBA": (0, 10), "XLogP": (None, 8)}}
    },
    "*cS(=O)(=O)Nc*": {
        "FXR": {"Domain": {"MW": (None, 900)}}
    },
    "[#6]~1~[#6]~[#6]~[#6]2~[#6](~[#6]1)~[#6]~[#6]~[#6]1~[#6]~[#6](~[#8])~[#6]~[#6]~[#6]~2~1": {
        "FXR": {"Domain": {"MW": (None, 900)}}
    },
    "[#6]~1~[#6]~[#6]~[#6]2~[#6](~[#6]1)~[#6]~[#6]~[#6]1~[#6]~[#6](~*~*~*~*~*~[#8])~[#6]~[#6]~[#6]~2~1": {
        "FXR": {"Domain": {"MW": (None, 900)}}
    },
    "[#6]1~[#6]~[#6]2~[#6](~[#6]~[#6]~[#6]3~[#6]~2~[#6]~[#6]~[#6]2~[#6]~[#6]~[#6]~[#6]~3~2)~[#6]~[#6]~1": {
        "GR": {"Domain": {"HBD": (0, 15), "MW": (300, 610), "HBA": (0, 10), "XLogP": (0, None)}},
        "PXR": {"Domain": {"HBD": (0, 15), "MW": (300, 610), "HBA": (0, 10), "XLogP": (0, None)}}
    },
    "Cc1ccc(F)cc1C": {
        "GR": {"Domain": {"HBD": (0, 15), "MW": (180, 610), "HBA": (0, 15), "XLogP": (-1, None)}}
    },
    "[#6]~1~[#6]~[#6](~[#6]~[#6]~[#8,#6,#7,#16]~1)-[#6]-c1ccccc1": {
        "ER": {"Domain": {"HBD": (0, 10), "MW": (140, 700), "HBA": (0, 15), "XLogP": (-2, None)}}
    },
    "*C#N": {
        "GR": {"Domain": {"HBD": (0, 15), "MW": (300, 610), "HBA": (0, 10), "XLogP": (0, None)}},
        "PXR": {"Domain": {"HBD": (0, 15), "MW": (300, 610), "HBA": (0, 10), "XLogP": (0, None)}}
    },
    "[#6]~1~[#6]~[#6]~[#6]~[#6]~2~[#6]~3~[#6]~[#6]~[#6]~[#6]~[#6]3~[#6]~[#6]~[#6]12": {
        "LXR": {"Domain": {"MW": (None, 750), "XLogP": (2, None)}}
    },
    "c1ccccc1CC(F)(F)F": {
        "LXR": {"Domain": {"MW": (None, 750), "XLogP": (2, None)}}
    },
    "a1aaaa1~*~*~*~*~*~*~c1ccccc1": {
        "LXR": {"Domain": {"MW": (None, 750), "XLogP": (2, None)}}
    },
    "a1aaaa1~*~*~*~c1ccccc1": {
        "LXR": {"Domain": {"MW": (None, 750), "XLogP": (2, None)}}
    },
    "a1aaaaa1~*~*~*~c1ccccc1": {
        "LXR": {"Domain": {"MW": (None, 750), "XLogP": (2, None)}}
    },
    "a1aaaaa1~*~*~c1ccccc1": {
        "LXR": {"Domain": {"MW": (None, 750), "XLogP": (2, None)}}
    },
    "a1aaaa1~*~*~*~*~c1ccccc1": {
        "LXR": {"Domain": {"MW": (None, 750), "XLogP": (2, None)}}
    },
    "a1aaaa1~*~*~c1ccccc1": {
        "LXR": {"Domain": {"MW": (None, 750), "XLogP": (2, None)}}
    },
    "O~Ca1aaaa1": {
        "LXR": {"Domain": {"MW": (None, 750), "XLogP": (2, None)}}
    },
    "C~1~C~C~C2~C(~C1)~C~C~C1~C~C~C~C~C~2~1": {
        "PPAR": {"Domain": {"MW": (None, 800)}}
    },
    "c1nc2cncnc2n1": {
        "PPAR": {"Domain": {"MW": (None, 800)}}
    },
    "a(a)a~*~*~a(a)a": {
        "PPAR": {"Domain": {"MW": (None, 800)}}
    },
    "*~[#6]~*~[#6]~*~[#6]~*~[#6]a1a([O,Cl,F,I,Br,N*])aaaa1": {
        "PPAR": {"Domain": {"MW": (None, 800)}}
    },
    "[#6]~*~[#6]~*~[#6]~*~[#6]~*~a1a([O,Cl,F,I,Br,N*])aaaa1": {
        "PPAR": {"Domain": {"MW": (None, 800)}}
    },
    "[#6]1~[#6]~[#6]2~[#6](~[#6]~[#6]~[#6]3~[#6]~2~[#6]~[#6]~[#6]2~[#6]~[#6]~[#6]~[#6]~3~2)~[#6]~[#6]~1": {
        "PXR": {"Domain": {"HBD": (0, 15), "MW": (300, 610), "HBA": (0, 10), "XLogP": (0, None)}}
    },
    "O~[#6]1~[#6]~[#6]~[#6]2~[#6](~[#6]~[#6]~[#6]3~[#6]~[#6]~[#6]~[#6]~[#6]~23)~[#6]~1": {
        "PXR": {"Domain": {"HBD": (0, 15), "MW": (300, 610), "HBA": (0, 10), "XLogP": (0, None)}}
    },
    "[#8,#6,#7,#16]~1~[#8,#6,#7,#16]~[#8,#6,#7,#16]~[#6](~[#8,#6,#7,#16]~[#8,#6,#7,#16]~1)-[#7,#8,#6,#16]-c1ccccc1": {
        "PXR": {"Domain": {"HBD": (0, 15), "MW": (300, 610), "HBA": (0, 10), "XLogP": (0, None)}}
    },
    "*C#N": {
        "GR": {"Domain": {"HBD": (0, 15), "MW": (300, 610), "HBA": (0, 10), "XLogP": (0, None)}},
        "PXR": {"Domain": {"HBD": (0, 15), "MW": (300, 610), "HBA": (0, 10), "XLogP": (0, None)}}
    },
    "O~[#6]~[#6]~[#7]~[#6]": {
        "RAR": {"Domain": {"MW": (None, 550)}}
    },
    "*[#6](~[#8])~[#6](~[#8])*": {
        "RAR": {"Domain": {"MW": (None, 550)}}
    },
}
# Create tabs
tab1, tab2, tab3 = st.tabs(["Predictor", "About", "Contact"])

HBD: [$\ge 0$, $\le 6$], MW: [$\ge 180$, $\le 900$], HBA: [$\ge 0$, $\le 10$], XLogP: [No Lower Limit, $\le 8$]
# Tab 2: About
with tab2:
    st.header("About Steatosis Predictor")
    st.markdown(
        """
        ### Purpose
        Steatosis Predictor queries a set of refined structural alerts, encoded as SMARTS patterns and a binary QSAR model to predict likelihood of steatosis uses the **RDKit** library, a powerful cheminformatics toolkit, to perform substructure searching based on **SMARTS (Simplified Molecular Input Line Entry System)** patterns.

        The application checks if the SMILES input format contains any of the predefined SMARTS patterns associated with potential steatosis-related Molecular Initiating Events (MIEs). It also assesses if the input molecule falls within the defined chemical property domain for each identified alert.

        ### Molecular Initiating Events (MIEs)
        The MIEs listed are based on current scientific understanding and literature linking specific structural features to the initiation of biological events relevant to steatosis. More information can be found here (10.1021/acs.chemrestox.5b00480)

        ### Libraries Used
        - **Streamlit:** For creating the interactive web application.
        - **RDKit:** For chemical informatics tasks, including SMILES parsing and SMARTS matching, and descriptor calculation.
        - **Pillow (PIL):** Implicitly used by Streamlit for image handling.
        """
    )

# Tab 3: Contact
with tab3:
    st.header("About Us")
    developers = ["Anish Gomatam", "James Firman", "Georgios Chrysochoou", "Mark Cronin"]
    st.write("Developed by:", ", ".join(developers))
