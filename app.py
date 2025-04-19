import streamlit as st
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import Descriptors
from rdkit.Chem import Lipinski

# Define SMARTS patterns and their associated MIEs with chemical property domains
smarts_mie_mapping = {
    "C(=C\\c1ccccc1)\c1ccccc1": {
        "MIEs": ["AhR"],
        "Domain": {"HBD": (0, 6), "MW": (180, 900), "HBA": (0, 10), "XLogP": (None, 8)}
    },
    "c1nc2ccccc2s1": {
        "MIEs": ["AhR"],
        "Domain": {"HBD": (0, 6), "MW": (180, 900), "HBA": (0, 10), "XLogP": (None, 8)}
    },
    "c1c*o*1": {
        "MIEs": ["AhR", "ER"],
        "Domain": {"HBD": (0, 10), "MW": (140, 700), "HBA": (0, 15), "XLogP": (-2, None)}
    },
    "[#7,#6,#8,#16]1[#7,#6,#8,#16][#7,#6,#8,#16][#7,#6,#8,#16]([#7,#6,#8,#16]1)-c1ccccc1": {
        "MIEs": ["AhR", "ER", "GR", "PXR"],
        "Domain": {"HBD": (0, 15), "MW": (300, 610), "HBA": (0, 10), "XLogP": (0, None)}
    },
    "[#8,#7,#6]~1~[#8,#7,#6]~[#8,#7,#6]~c2ccccc2~[#8,#7,#6]~1": {
        "MIEs": ["AhR"],
        "Domain": {"HBD": (0, 6), "MW": (180, 900), "HBA": (0, 10), "XLogP": (None, 8)}
    },
    "[#6]-[#7]-c1ccccc1-[#9,#17]": {
        "MIEs": ["AhR"],
        "Domain": {"HBD": (0, 6), "MW": (180, 900), "HBA": (0, 10), "XLogP": (None, 8)}
    },
    "*cS(=O)(=O)Nc*": {
        "MIEs": ["FXR"],
        "Domain": {"MW": (None, 900)}
    },
    "[#6]~1~[#6]~[#6]~[#6]2~[#6](~[#6]1)~[#6]~[#6]~[#6]1~[#6]~[#6](~[#8])~[#6]~[#6]~[#6]~2~1": {
        "MIEs": ["FXR"],
        "Domain": {"MW": (None, 900)}
    },
    "[#6]~1~[#6]~[#6]~[#6]2~[#6](~[#6]1)~[#6]~[#6]~[#6]1~[#6]~[#6](~*~*~*~*~*~[#8])~[#6]~[#6]~[#6]~2~1": {
        "MIEs": ["FXR"],
        "Domain": {"MW": (None, 900)}
    },
    "[#6]1~[#6]~[#6]2~[#6](~[#6]~[#6]~[#6]3~[#6]~2~[#6]~[#6]~[#6]2~[#6]~[#6]~[#6]~[#6]~3~2)~[#6]~[#6]~1": {
        "MIEs": ["GR", "PXR"],
        "Domain": {"HBD": (0, 15), "MW": (300, 610), "HBA": (0, 10), "XLogP": (0, None)}
    },
    "Cc1ccc(F)cc1C": {
        "MIEs": ["GR"],
        "Domain": {"HBD": (0, 15), "MW": (180, 610), "HBA": (0, 15), "XLogP": (-1, None)}
    },
    "[#6]~1~[#6]~[#6](~[#6]~[#6]~[#8,#6,#7,#16]~1)-[#6]-c1ccccc1": {
        "MIEs": ["ER"],
        "Domain": {"HBD": (0, 10), "MW": (140, 700), "HBA": (0, 15), "XLogP": (-2, None)}
    },
    "*C#N": {
        "MIEs": ["GR", "PXR"],
        "Domain": {"HBD": (0, 15), "MW": (300, 610), "HBA": (0, 10), "XLogP": (0, None)}
    },
    "[#6]~1~[#6]~[#6]~[#6]~[#6]~2~[#6]~3~[#6]~[#6]~[#6]~[#6]~[#6]3~[#6]~[#6]~[#6]12": {
        "MIEs": ["LXR"],
        "Domain": {"MW": (None, 750), "XLogP": (2, None)}
    },
    "c1ccccc1CC(F)(F)F": {
        "MIEs": ["LXR"],
        "Domain": {"MW": (None, 750), "XLogP": (2, None)}
    },
    "a1aaaa1~*~*~*~*~*~*~c1ccccc1": {
        "MIEs": ["LXR"],
        "Domain": {"MW": (None, 750), "XLogP": (2, None)}
    },
    "a1aaaa1~*~*~*~c1ccccc1": {
        "MIEs": ["LXR"],
        "Domain": {"MW": (None, 750), "XLogP": (2, None)}
    },
    "a1aaaaa1~*~*~*~c1ccccc1": {
        "MIEs": ["LXR"],
        "Domain": {"MW": (None, 750), "XLogP": (2, None)}
    },
    "a1aaaaa1~*~*~c1ccccc1": {
        "MIEs": ["LXR"],
        "Domain": {"MW": (None, 750), "XLogP": (2, None)}
    },
    "a1aaaa1~*~*~*~*~c1ccccc1": {
        "MIEs": ["LXR"],
        "Domain": {"MW": (None, 750), "XLogP": (2, None)}
    },
    "a1aaaa1~*~*~c1ccccc1": {
        "MIEs": ["LXR"],
        "Domain": {"MW": (None, 750), "XLogP": (2, None)}
    },
    "O~Ca1aaaa1": {
        "MIEs": ["LXR"],
        "Domain": {"MW": (None, 750), "XLogP": (2, None)}
    },
    "C~1~C~C~C2~C(~C1)~C~C~C1~C~C~C~C~C~2~1": {
        "MIEs": ["PPAR"],
        "Domain": {"MW": (None, 800)}
    },
    "c1nc2cncnc2n1": {
        "MIEs": ["PPAR"],
        "Domain": {"MW": (None, 800)}
    },
    "a(a)a~*~*~a(a)a": {
        "MIEs": ["PPAR"],
        "Domain": {"MW": (None, 800)}
    },
    "*~[#6]~*~[#6]~*~[#6]~*~[#6]a1a([O,Cl,F,I,Br,N*])aaaa1": {
        "MIEs": ["PPAR"],
        "Domain": {"MW": (None, 800)}
    },
    "[#6]~*~[#6]~*~[#6]~*~[#6]~*~a1a([O,Cl,F,I,Br,N*])aaaa1": {
        "MIEs": ["PPAR"],
        "Domain": {"MW": (None, 800)}
    },
    "[#6]1~[#6]~[#6]2~[#6](~[#6]~[#6]~[#6]3~[#6]~2~[#6]~[#6]~[#6]2~[#6]~[#6]~[#6]~[#6]~3~2)~[#6]~[#6]~1": {
        "MIEs": ["PXR"],
        "Domain": {"HBD": (0, 15), "MW": (300, 610), "HBA": (0, 10), "XLogP": (0, None)}
    },
    "O~[#6]1~[#6]~[#6]~[#6]2~[#6](~[#6]~[#6]~[#6]3~[#6]~[#6]~[#6]~[#6]~[#6]~23)~[#6]~1": {
        "MIEs": ["PXR"],
        "Domain": {"HBD": (0, 15), "MW": (300, 610), "HBA": (0, 10), "XLogP": (0, None)}
    },
    "[#8,#6,#7,#16]~1~[#8,#6,#7,#16]~[#8,#6,#7,#16]~[#6](~[#8,#6,#7,#16]~[#8,#6,#7,#16]~1)-[#7,#8,#6,#16]-c1ccccc1": {
        "MIEs": ["PXR"],
        "Domain": {"HBD": (0, 15), "MW": (300, 610), "HBA": (0, 10), "XLogP": (0, None)}
    },
    "*C#N": {
        "MIEs": ["GR", "PXR"],
        "Domain": {"HBD": (0, 15), "MW": (300, 610), "HBA": (0, 10), "XLogP": (0, None)}
    },
    "O~[#6]~[#6]~[#7]~[#6]": {
        "MIEs": ["RAR"],
        "Domain": {"MW": (None, 550)}
    },
    "*[#6](~[#8])~[#6](~[#8])*": {
        "MIEs": ["RAR"],
        "Domain": {"MW": (None, 550)}
    },
}

# Create tabs
tab1, tab2, tab3 = st.tabs(["Predictor", "About", "Contact"])

# Tab 1: Predictor
with tab1:
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

    # Display molecule and SMARTS matches with MIEs and domain check
    if mol:
        st.subheader("Molecule Structure")
        st.image(Draw.MolToImage(mol, size=(300, 300)))

        # Calculate properties for domain check
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        hbd = Lipinski.NumHDonors(mol)
        hba = Lipinski.NumHAcceptors(mol)

        st.subheader("SMARTS Matching Results with Associated MIEs and Domain Check:")
        results = []
        for smarts, data in smarts_mie_mapping.items():
            pattern = Chem.MolFromSmarts(smarts)
            if mol.HasSubstructMatch(pattern):
                mies = data["MIEs"]
                domain = data.get("Domain")
                domain_check = {}

                if domain:
                    within_domain = True
                    for prop, (min_val, max_val) in domain.items():
                        mol_prop_value = None
                        if prop == "MW":
                            mol_prop_value = mw
                        elif prop == "XLogP":
                            mol_prop_value = logp
                        elif prop == "HBD":
                            mol_prop_value = hbd
                        elif prop == "HBA":
                            mol_prop_value = hba

                        if mol_prop_value is not None:
                            lower_bound_met = (min_val is None) or (mol_prop_value >= min_val)
                            upper_bound_met = (max_val is None) or (mol_prop_value <= max_val)
                            domain_check[prop] = (lower_bound_met and upper_bound_met, f"[{min_val if min_val is not None else '-∞'}, {max_val if max_val is not None else '∞'}]")
                            if not (lower_bound_met and upper_bound_met):
                                within_domain = False
                        else:
                            domain_check[prop] = (False, "Property not calculated")
                            within_domain = False
                    results.append({"SMARTS": smarts, "MIE(s)": ", ".join(mies), "Domain": domain, "Within Domain": within_domain, "Domain Check": domain_check})
                else:
                    results.append({"SMARTS": smarts, "MIE(s)": ", ".join(mies), "Domain": "Not Available", "Within Domain": "N/A", "Domain Check": "N/A"})

        if results:
            # Format the 'Domain' and 'Domain Check' columns for better readability
            formatted_results = []
            for res in results:
                formatted_domain = "Not Available"
                if res["Domain"] and res["Domain"] != "Not Available":
                    formatted_domain = ", ".join([f"{k}: [{v[0] if v[0] is not None else '-∞'}, {v[1] if v[1] is not None else '∞'}]" for k, v in res["Domain"].items()])

                formatted_domain_check = "N/A"
                if res["Domain Check"] != "N/A":
                    formatted_domain_check = ", ".join([f"{k}: {v[0]} ({v[1]})" for k, v in res["Domain Check"].items()])

                formatted_results.append({
                    "SMARTS": res["SMARTS"],
                    "MIE(s)": res["MIE(s)"],
                    "Domain": formatted_domain,
                    "Within Domain": "Yes" if res["Within Domain"] is True else ("No" if res["Within Domain"] is False else "N/A"),
                    "Domain Check": formatted_domain_check,
                })
            st.dataframe(formatted_results)
        else:
            st.info("No matching SMARTS found for the given molecule.")
    else:
        st.info("Please enter a SMILES string.")

# Tab 2: About
with tab2:
    st.header("About Steatosis Predictor")
    st.markdown(
        ""
        ### Purpose
        Steatosis Predictor queries a set of refined structural alerts, encoded as SMARTS patterns and a binary QSAR model to predict likelihood of ste
