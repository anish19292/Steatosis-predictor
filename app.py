import streamlit as st
from rdkit import Chem
from rdkit.Chem import Draw

# Define SMARTS patterns
smarts_list = [
    "[#7,#6,#8,#16]1[#7,#6,#8,#16][#7,#6,#8,#16][#7,#6,#8,#16]([#7,#6,#8,#16]1)-c1ccccc1",
    "[#6]~1~[#6]~[#6](~[#6]~[#6]~[#8,#6,#7,#16]~1)-[#6]-c1ccccc1"
]

pattern_names = [
    "SMARTS 1",
    "SMARTS 2"
]

# Page title
st.title("Steatosis SMARTS Matcher")
st.write("Enter a SMILES string to check for steatosis structural alerts.")

# Input: SMILES string
smiles_input = st.text_input("Enter SMILES:", "CC1CCCCC1")

# Convert SMILES to RDKit Mol
mol = None
if smiles_input:
    try:
        mol = Chem.MolFromSmiles(smiles_input)
    except:
        st.error("Invalid SMILES input.")

# Display molecule and SMARTS matches
if mol:
    st.subheader("Molecule Structure")
    st.image(Draw.MolToImage(mol, size=(300, 300)))

    st.subheader("SMARTS Matching Results")
    results = []
    for name, smarts in zip(pattern_names, smarts_list):
        pattern = Chem.MolFromSmarts(smarts)
        if mol.HasSubstructMatch(pattern):
            results.append(f"✅ {name} matched.")
        else:
            results.append(f"❌ {name} did not match.")
    
    for result in results:
        st.write(result)
