import streamlit as st
from rdkit import Chem
from rdkit.Chem import Draw

# Define SMARTS patterns and their associated MIEs
smarts_mie_mapping = {
    "C(=C\\c1ccccc1)\c1ccccc1": ["AhR"],
    "c1nc2ccccc2s1": ["AhR"],
    "c1c*o*1": ["AhR", "ER"],
    "[#7,#6,#8,#16]1[#7,#6,#8,#16][#7,#6,#8,#16][#7,#6,#8,#16]([#7,#6,#8,#16]1)-c1ccccc1": ["AhR", "ER", "GR", "PXR"],
    "[#8,#7,#6]~1~[#8,#7,#6]~[#8,#7,#6]~c2ccccc2~[#8,#7,#6]~1": ["AhR"],
    "[#6]-[#7]-c1ccccc1-[#9,#17]": ["AhR"],
    "*cS(=O)(=O)Nc*": ["FXR"],
    "[#6]~1~[#6]~[#6]~[#6]2~[#6](~[#6]1)~[#6]~[#6]~[#6]1~[#6]~[#6](~[#8])~[#6]~[#6]~[#6]~2~1": ["FXR"],
    "[#6]~1~[#6]~[#6]~[#6]2~[#6](~[#6]1)~[#6]~[#6]~[#6]1~[#6]~[#6](~*~*~*~*~*~[#8])~[#6]~[#6]~[#6]~2~1": ["FXR"],
    "[#6]1~[#6]~[#6]2~[#6](~[#6]~[#6]~[#6]3~[#6]~2~[#6]~[#6]~[#6]2~[#6]~[#6]~[#6]~[#6]~3~2)~[#6]~[#6]~1": ["GR", "PXR"],
    "Cc1ccc(F)cc1C": ["GR"],
    "*C#N": ["GR", "PXR"],
    "[#6]~1~[#6]~[#6]~[#6]~[#6]~2~[#6]~3~[#6]~[#6]~[#6]~[#6]~[#6]3~[#6]~[#6]~[#6]12": ["LXR"],
    "c1ccccc1CC(F)(F)F": ["LXR"],
    "a1aaaa1~*~*~*~*~*~*~c1ccccc1": ["LXR"],
    "a1aaaa1~*~*~*~c1ccccc1": ["LXR"],
    "a1aaaaa1~*~*~*~c1ccccc1": ["LXR"],
    "a1aaaaa1~*~*~c1ccccc1": ["LXR"],
    "a1aaaa1~*~*~*~*~c1ccccc1": ["LXR"],
    "a1aaaa1~*~*~c1ccccc1": ["LXR"],
    "O~Ca1aaaa1": ["LXR"],
    "C~1~C~C~C2~C(~C1)~C~C~C1~C~C~C~C~C~2~1": ["PPAR"],
    "c1nc2cncnc2n1": ["PPAR"],
    "a(a)a~*~*~a(a)a": ["PPAR"],
    "*~[#6]~*~[#6]~*~[#6]~*~[#6]a1a([O,Cl,F,I,Br,N*])aaaa1": ["PPAR"],
    "[#6]~*~[#6]~*~[#6]~*~[#6]~*~a1a([O,Cl,F,I,Br,N*])aaaa1": ["PPAR"],
    "O~[#6]1~[#6]~[#6]~[#6]2~[#6](~[#6]~[#6]~[#6]3~[#6]~[#6]~[#6]~[#6]~[#6]~23)~[#6]~1": ["PXR"],
    "[#8,#6,#7,#16]~1~[#8,#6,#7,#16]~[#8,#6,#7,#16]~[#6](~[#8,#6,#7,#16]~[#8,#6,#7,#16]~1)-[#7,#8,#6,#16]-c1ccccc1": ["PXR"],
    "O~[#6]~[#6]~[#7]~[#6]": ["RAR"],
    "*[#6](~[#8])~[#6](~[#8])*": ["RAR"],
}

# Page title
st.title("Steatosis Predictor")
st.write("Enter a SMILES string to check for steatosis structural alerts and associated Molecular Initiating Events (MIEs).")

# Input: SMILES string
smiles_input = st.text_input("Enter SMILES:", "CC1CCCCC1")

# Convert SMILES to RDKit Mol
mol = None
if smiles_input:
    try:
        mol = Chem.MolFromSmiles(smiles_input)
    except:
        st.error("Invalid SMILES input.")

# Display molecule and SMARTS matches with MIEs
if mol:
    st.subheader("Molecule Structure")
    st.image(Draw.MolToImage(mol, size=(300, 300)))

    st.subheader("SMARTS Matching Results with Associated MIEs:")
    results = []
    for smarts, mies in smarts_mie_mapping.items():
        pattern = Chem.MolFromSmarts(smarts)
        if mol.HasSubstructMatch(pattern):
            results.append({"SMARTS": smarts, "MIE(s)": ", ".join(mies)})

    if results:
        st.dataframe(results)
    else:
        st.info("No matching SMARTS found for the given molecule.")
else:
    st.info("Please enter a SMILES string.")
