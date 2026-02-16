# ROCK-DIC
ROCk Kinematics via Digital Image Correlation
RockDIC: Rockfall Monitoring & Analysis via Digital Image CorrelationRockDIC is an advanced software solution designed for monitoring unstable rock faces and identifying potential rockfall hazards using Digital Image Correlation (DIC) techniques. It allows geologists and engineers to detect millimetric displacements and structural precursors to slope failure from photographic sequences.🌟 Key FeaturesOptical Pre-processing: Automatic image registration (alignment) and enhancement filters (contrast, noise reduction) to ensure data integrity.ROI & Masking Management: Intuitive tools to define Regions of Interest or exclude problematic areas such as vegetation or moving shadows.Multi-Model DIC Analysis: Support for various correlation algorithms tailored to different rock textures and lighting conditions.Manual Review & Validation: A dedicated interface to inspect displacement vectors and manually validate deformation zones.Automated Reporting: Generate comprehensive reports featuring displacement maps, time-series plots, and exportable statistics.🛠️ Technical WorkflowThe software follows a rigorous scientific pipeline:Alignment: Image registration and homography correction.Masking: Selection of analysis areas.Correlation: Calculation of $u$ and $v$ displacement fields.Validation: Outlier filtering and human-in-the-loop verification.Reporting: Final PDF/Excel data export.⚙️ InstallationBash# Clone the repository
git clone https://github.com/your-username/rock-dic.git

# Enter the directory
cd rock-dic

# Install dependencies
pip install -r requirements.txt

# Launch the application
python main.py
📖 Quick StartLoad Images: Import your time-lapse sequence of the rock face.Enhance: Apply the "CLAHE" filter to sharpen rock textures.Analyze: Select your correlation parameters and run the solver.Validate: Use the "Validation Brush" to confirm detected movement zones.Report: Generate your detailed technical report with one click.📄 LicenseThis project is licensed under the MIT License - see the LICENSE file for details.
