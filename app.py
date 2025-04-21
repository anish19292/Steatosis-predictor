import streamlit as st
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import Descriptors
from rdkit.Chem import Lipinski
from rdkit.Chem import AllChem, MACCSkeys, RDKFingerprint, LayeredFingerprint, PatternFingerprint
import pickle
import numpy as np

def load_classifier():
    with open("classifier.pkl", "rb") as f:
        model = pickle.load(f)
    return model

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
    "O~[#6]~[#6]~[#7]~[#6]": {
        "RAR": {"Domain": {"MW": (None, 550)}}
    },
    "*[#6](~[#8])~[#6](~[#8])*": {
        "RAR": {"Domain": {"MW": (None, 550)}}
    },
}

# Create tabs
tab1, tab2, tab3 = st.tabs(["Predictor", "About", "Contact"])
# Tab 1: Predictor
with tab1:
    st.title("Steatosis Predictor")
    st.write("Enter a SMILES string to check for steatosis structural alerts and their associated Molecular Initiating Events (MIEs) with MIE-specific chemical property domain assessment.")

    # Input: SMILES string
    smiles_input = st.text_input("Enter SMILES:", "CC1CCCCC1")

    # Function to compute fingerprints silently
    def compute_fingerprints(mol):
        fingerprints = {}

        # RDKit fingerprints with varying lengths
        for size in [93, 204, 292, 405, 690, 718, 926]:
            fp = RDKFingerprint(mol, fpSize=size)
            fingerprints[f"RDKit_fp_{size}"] = list(fp)

        # Layered fingerprint
        layered_fp = LayeredFingerprint(mol, fpSize=109)
        fingerprints["Layered_fp_109"] = list(layered_fp)

        # Pattern fingerprint (truncate to 779 bits)
        pattern_fp = PatternFingerprint(mol)
        fingerprints["Pattern_fp_779"] = list(pattern_fp)[:779]

        return fingerprints

    # Convert SMILES to RDKit Mol
    mol = None
    if smiles_input:
        try:
            mol = Chem.MolFromSmiles(smiles_input)
        except:
            st.error("Invalid SMILES input.")

    if mol:
        st.subheader("Molecule Structure")
        st.image(Draw.MolToImage(mol, size=(300, 300)))

        # Calculate fingerprints silently (not displayed)
        fingerprints = compute_fingerprints(mol)

        # Flatten the fingerprints into a single feature array
        fingerprint_features = []
        for size in [93, 204, 292, 405, 690, 718, 926, 109, 779]:
            fp_key = f"RDKit_fp_{size}" if size != 109 and size != 779 else f"Layered_fp_109" if size == 109 else f"Pattern_fp_779"
            fingerprint_features.extend(fingerprints[fp_key])

        # Convert fingerprint features to numpy array
        feature_vector = np.array(fingerprint_features).reshape(1, -1)  # Reshaping to match classifier input

        # Load the model and make a prediction
        model = load_classifier()

        # Predict using the model
        prediction = model.predict(feature_vector)

        # Display the result
        if prediction:
            st.subheader("Prediction Result:")
            st.write(f"Predicted Class: {prediction[0]}")
        else:
            st.error("Prediction could not be made.")
    else:
        st.info("Please enter a valid SMILES string.")

        # Property-based domain analysis
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        hbd = Lipinski.NumHDonors(mol)
        hba = Lipinski.NumHAcceptors(mol)

        st.subheader("Structural Alert Analysis with MIE-Specific Domain Check:")
        results = []
        for smarts, mie_data in smarts_mie_mapping.items():
            pattern = Chem.MolFromSmarts(smarts)
            if mol.HasSubstructMatch(pattern):
                for mie, data in mie_data.items():
                    domain = data.get("Domain")
                    within_domain = True
                    formatted_domain = "Not Available"

                    if domain:
                        domain_strings = []
                        for prop, (min_val, max_val) in domain.items():
                            lower_bound = f">= {min_val}" if min_val is not None else ""
                            upper_bound = f"<= {max_val}" if max_val is not None else ""
                            range_str = ""
                            if lower_bound and upper_bound:
                                range_str = f"[{lower_bound}, {upper_bound}]"
                            elif lower_bound:
                                range_str = f"[{lower_bound}]"
                            elif upper_bound:
                                range_str = f"[{upper_bound}]"
                            else:
                                range_str = "No Limit"
                            domain_strings.append(f"{prop}: {range_str}")
                        formatted_domain = ", ".join(domain_strings)

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
                                if not (lower_bound_met and upper_bound_met):
                                    within_domain = False
                            else:
                                within_domain = False

                    results.append({
                        "SMARTS": smarts,
                        "MIE": mie,
                        "Domain": formatted_domain if domain else "Not Available",
                        "Within Domain": "Yes" if within_domain is True else ("No" if within_domain is False else "N/A"),
                    })

        if results:
    st.subheader("Matching Alerts and MIE-Specific Domain Check:")
    st.dataframe(results)
else:
    st.info("No matching structural alerts found for the given molecule.")
    
if not mol:
    st.info("Please enter a valid SMILES string.")

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
