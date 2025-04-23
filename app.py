import streamlit as st
import pickle
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import Descriptors
from rdkit.Chem import Lipinski
from rdkit.Chem import AllChem, MACCSkeys, RDKFingerprint, LayeredFingerprint, PatternFingerprint
import numpy as np
import pandas as pd

# Function to load the classifier and feature names
def load_model():
    try:
        with open('classifier.pkl', 'rb') as f:
            model_data = pickle.load(f)

        # Check that the keys exist in the loaded data
        if 'classifier' in model_data and 'feat_names' in model_data:
            loaded_classifier = model_data['classifier']
            loaded_feat_names = model_data['feat_names']
            print(f"Loaded classifier type: {type(loaded_classifier)}")  # Print to verify type
            return loaded_classifier, loaded_feat_names
        else:
            raise KeyError("Missing keys: 'classifier' or 'feat_names' in the model data.")
    except FileNotFoundError:
        st.error("Model file 'classifier.pkl' not found.")
    except KeyError as e:
        st.error(f"Key error: {e}")
    except Exception as e:
        st.error(f"An error occurred while loading the model: {e}")
    return None, None

# Load the model
loaded_classifier, loaded_feat_names = load_model()

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

# Create empty space columns on the sides and narrower image columns in the center
spacer1, col1, spacer2, col2, spacer3 = st.columns([0.1, 0.4, 0.1, 0.4, 0.1])

with col1:
    st.image("LJMU image.gif", use_column_width=True)

with col2:
    st.image("risk-hunter-og.png", use_column_width=True)

# Then your tabs
tab1, tab2, tab3, tab4 = st.tabs(["Predictor", "About", "Contact", "Acknowledgement"])

# Tab 1: Predictor
with tab1:
    st.title("Steatosis Predictor")
    st.write("Enter a SMILES string to check for steatosis structural alerts and their associated Molecular Initiating Events (MIEs) with MIE-specific chemical property domain assessment.")

    # Input: SMILES string
    smiles_input = st.text_input("Enter SMILES:", "CC1CCCCC1")

    # Convert SMILES to RDKit Mol
    mol = None
    if smiles_input:
        try:
            mol = Chem.MolFromSmiles(smiles_input)
            if mol is None:
                st.error("Invalid SMILES input.")
        except Exception as e:
            st.error(f"Error processing SMILES: {e}")

    if mol:
        # Display the molecule structure
        st.subheader("Molecule Structure")
        st.image(Draw.MolToImage(mol, size=(300, 300)))

        # Property-based domain analysis
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        hbd = Lipinski.NumHDonors(mol)
        hba = Lipinski.NumHAcceptors(mol)

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
                        "Within Domain": "Yes" if within_domain else "No",
                    })

        if results:
            st.subheader("Matching Alerts and MIE-Specific Domain Check:")
            st.dataframe(results)
        else:
            st.info("No matching structural alerts found for the given molecule.")

        # ---- FINGERPRINTS & PREDICTION (only in Tab 1) ----
        if loaded_classifier:
            st.subheader("RDKit Fingerprint Calculation:")
            rdkit_fp = RDKFingerprint(mol, fpSize=1024)
            fp_array = np.array(rdkit_fp)

            fingerprint_names = [
                "RDKit fingerprints93", "RDKit fingerprints204", "RDKit fingerprints292",
                "RDKit fingerprints405", "RDKit fingerprints690", "RDKit fingerprints718", "RDKit fingerprints926"
            ]
            selected_fingerprints = [
                fp_array[93], fp_array[204], fp_array[292],
                fp_array[405], fp_array[690], fp_array[718], fp_array[926]
            ]

            input_data = {name: value for name, value in zip(fingerprint_names, selected_fingerprints)}
            input_df = pd.DataFrame([input_data])

            st.write("Selected Fingerprints (Named Bits and Values):")
            st.dataframe(input_df)

            prediction = loaded_classifier.predict(input_df)

            st.subheader("Binary QSAR Prediction:")
            if prediction[0] == 1:
                st.success("The given molecule is likely to be steatotic")
            else:
                st.warning("The given molecule is likely to be steatotic")

# Tab 2: About
with tab2:
    st.header("About Steatosis Predictor")
    st.markdown(
        """
        ### Purpose
        Steatosis Predictor queries a set of refined structural alerts, encoded as SMARTS patterns and a binary QSAR model to predict likelihood of steatosis.

        The application checks if the SMILES input format contains any of the predefined SMARTS patterns associated with potential steatosis-related Molecular Initiating Events (MIEs). It also assesses if the input molecule falls within the defined chemical property domain for each identified alert.

        ### Molecular Initiating Events (MIEs)
        The MIEs listed are based on current scientific understanding and literature linking specific structural features to the initiation of biological events relevant to steatosis. These often involve interaction with nuclear receptors that regulate gene expression related to lipid metabolism and inflammation.

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
        | Retinoid X receptor                            | RXR          | —                            |

        More information can be found here: [10.1021/acs.chemrestox.5b00480](https://doi.org/10.1021/acs.chemrestox.5b00480)
        """
    )

with tab3:
    st.header("About Us")
    st.markdown(
        """
        <div style="font-size:16px; line-height:1.8">

        <p><strong>Anish Gomatam</strong> – Postdoctoral Researcher<br>
        Email: <a href="mailto:A.Gomatam@ljmu.ac.uk">A.Gomatam@ljmu.ac.uk</a></p>

        <p><strong>James Firman</strong> – Postdoctoral Researcher<br>
        Email: <a href="mailto:J.W.Firman@ljmu.ac.uk">J.W.Firman@ljmu.ac.uk</a></p>

        <p><strong>Georgios Chrysochoou</strong> – Postdoctoral Researcher<br>
        Email: <a href="mailto:G.Chrysochoou@ljmu.ac.uk">G.Chrysochoou@ljmu.ac.uk</a></p>

        <p><strong>Prof. Mark Cronin</strong> – Principal Investigator<br>
        Email: <a href="mailto:M.T.Cronin@ljmu.ac.uk">M.T.Cronin@ljmu.ac.uk</a></p>

        <p><strong>Affiliation:</strong><br>
        School of Pharmacy and Biomolecular Sciences,<br>
        Liverpool John Moores University,<br>
        Byrom Street, Liverpool L3 3AF, United Kingdom</p>

        </div>
        """,
        unsafe_allow_html=True
    )

# Add Acknowledgement tab
with tab4:
    st.title("Acknowledgement")
    st.write(
        """
        This project receives funding from the European Union's Horizon 2020 Research and Innovation programme under Grant Agreement No. 964537 (RISK-HUNT3R), 
        and it is part of the ASPIS cluster.
        """
    )

