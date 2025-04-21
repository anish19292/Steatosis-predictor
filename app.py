from rdkit.Chem import RDKFingerprint, LayeredFingerprint, PatternFingerprint

# Function to compute fingerprints
def compute_all_fingerprints(mol):
    fingerprints = {}

    # RDKit fingerprints (size 1024, with expanded bit vector)
    rdkit_fp = RDKFingerprint(mol, fpSize=1024, useFeatures=True)  # Expanded bit vector
    fingerprints["RDKit_fp_1024"] = list(rdkit_fp)
    print(f"RDKit_fp_1024: Length = {len(fingerprints['RDKit_fp_1024'])}")  # Check length

    # Layered fingerprint (default size: 2048 bits)
    layered_fp = LayeredFingerprint(mol)
    fingerprints["Layered_fp_2048"] = list(layered_fp)
    print(f"Layered_fp_2048: Length = {len(fingerprints['Layered_fp_2048'])}")  # Check length

    # Pattern fingerprint (default size: 2048 bits)
    pattern_fp = PatternFingerprint(mol)
    fingerprints["Pattern_fp_2048"] = list(pattern_fp)
    print(f"Pattern_fp_2048: Length = {len(fingerprints['Pattern_fp_2048'])}")  # Check length

    return fingerprints

# Assuming mol is a valid molecule object
if mol:
    fingerprints = compute_all_fingerprints(mol)

    # Display the calculated fingerprints
    st.write("Calculated Fingerprints:")
    for key, fp in fingerprints.items():
        st.write(f"{key}: {fp[:10]}... (showing first 10 bits)")
else:
    st.error("Invalid molecule. Please check the SMILES string.")

    # 2. Flatten all fingerprints into a single dictionary
    flat_features = {}
    for fp_type, bitlist in all_fps.items():
        for i, bit in enumerate(bitlist):
            flat_features[f"{fp_type}_{i}"] = bit

    # 3. Filter only model-relevant features
    try:
        model_input_vector = [flat_features[feat] for feat in loaded_feat_names]
    except KeyError as e:
        st.error(f"Missing feature in fingerprint: {e}")
        st.stop()

    # 4. Convert to NumPy array and reshape
    model_input_array = np.array(model_input_vector).reshape(1, -1)

    # 5. Predict using classifier
    prediction = loaded_classifier.predict(model_input_array)

    # 6. Display result
    st.subheader("Prediction Result")
    st.write("⚠️ **Predicted: Steatosis**" if prediction[0] == 1 else "✅ **Predicted: Non-Steatosis**")



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
