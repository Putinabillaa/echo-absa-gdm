import csv
import subprocess
import os
import glob
import shutil
from nicegui import ui
import threading
import queue
import time

class PipelineGUI:
    def __init__(self):
        self.log_queue = queue.Queue()
        self.is_running = False
        self.uploaded_files = {}
        self.timings = {}

    def run_cmd(self, cmd, description):
        """Run command and log output with timing"""
        self.log(f"\n[INFO] Running: {description}")
        self.log(f"[CMD] {' '.join(cmd)}")
        start = time.time()
        try:
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            elapsed = time.time() - start
            self.timings[description] = elapsed

            if result.returncode != 0:
                self.log(f"[ERROR] {description} failed:\n{result.stderr}")
                return False
            if result.stdout.strip():
                self.log(result.stdout.strip())
            self.log(f"[DONE] {description} complete. (Time: {elapsed:.2f}s)")
            return True
        except Exception as e:
            self.log(f"[ERROR] Failed to run {description}: {str(e)}")
            return False

    def log(self, message):
        self.log_queue.put(message)

    def handle_file_upload(self, e, file_type):
        """Handle file upload and store in temporary location"""
        try:
            uploads_dir = "uploads"
            os.makedirs(uploads_dir, exist_ok=True)
            upload_path = os.path.join(uploads_dir, e.name)
            with open(upload_path, 'wb') as f:
                f.write(e.content.read())
            self.uploaded_files[file_type] = upload_path
            self.log(f"[INFO] Uploaded {file_type}: {e.name}")
        except Exception as ex:
            self.log(f"[ERROR] Failed to upload {file_type}: {str(ex)}")

    def create_absa_input_from_community(self, community_csv_path, absa_input_path):
        start = time.time()
        try:
            with open(community_csv_path, newline="", encoding="utf-8") as infile, \
                 open(absa_input_path, "w", newline="", encoding="utf-8") as outfile:

                reader = csv.DictReader(infile)
                fieldnames = ["id", "text"]
                writer = csv.DictWriter(outfile, fieldnames=fieldnames)
                writer.writeheader()
                for row in reader:
                    if "id" not in row or "text" not in row:
                        self.log("[ERROR] community_input CSV must contain 'id' and 'text'")
                        return False
                    writer.writerow({"id": row["id"], "text": row["text"]})
            elapsed = time.time() - start
            self.timings["Create ABSA Input"] = elapsed
            self.log(f"[DONE] Created ABSA input. (Time: {elapsed:.2f}s)")
            return True
        except Exception as e:
            self.log(f"[ERROR] Failed to create ABSA input: {str(e)}")
            return False

    def run_pipeline(self, config):
        """Run pipeline sequentially like CLI version"""
        self.is_running = True
        self.timings = {}
        total_start = time.time()
        try:
            os.makedirs(config['workdir'], exist_ok=True)
            absa_input_csv = os.path.join(config['workdir'], "absa_input.csv")
            
            # Community input
            community_input_path = self.uploaded_files.get('community_input')
            if not community_input_path or not os.path.exists(community_input_path):
                self.log("[ERROR] Community input file not found")
                return
                
            if not self.create_absa_input_from_community(community_input_path, absa_input_csv):
                return

            aspect_output_folder = os.path.join(config['workdir'], "aspect_out")
            aspect_file_path = self.uploaded_files.get('absa_aspect')
            if not aspect_file_path or not os.path.exists(aspect_file_path):
                self.log("[ERROR] ABSA aspect file not found")
                return
                
            if config['model'] == "gemini":
                if not self.run_cmd([
                    "gemini",
                    "--input", absa_input_csv,
                    "--aspects", aspect_file_path,
                    "--output", aspect_output_folder,
                    "--model", config['model_name'],
                    "--batch_size", str(int(config['batch_size']))
                ], "Component 1: Gemini"):
                    return
            elif config['model'] == "dp":
                lexicon_path = self.uploaded_files.get('lexicon_file')
                if not lexicon_path or not os.path.exists(lexicon_path):
                    self.log("[ERROR] Lexicon file required for DP model")
                    return
                    
                if not self.run_cmd([
                    "dp",
                    "--input", absa_input_csv,
                    "--aspects", aspect_file_path,
                    "--lexicon", lexicon_path,
                    "--output", aspect_output_folder,
                    "--vector_mode", config['vector_mode'],
                    "--processor", config['model_name']
                ], "Component 1: Double Propagation (DP)"):
                    return

            # ABSA output
            pattern = os.path.join(aspect_output_folder, "*.csv")
            matching_files = glob.glob(pattern)
            absa_output_csv = matching_files[0] if matching_files else None
            if not absa_output_csv:
                self.log("[ERROR] No ABSA output file found")
                return

            # Community detection
            community_output_csv = os.path.join(config['workdir'], "community_out.csv")
            graph_output_gml = os.path.join(config['workdir'], "community_graph.gml")
            already_has_community = False
            try:
                with open(community_input_path, newline="", encoding="utf-8") as infile:
                    reader = csv.DictReader(infile)
                    if "community" in reader.fieldnames:
                        already_has_community = True
                        self.log("[INFO] 'community' column found in input, skipping community detection.")
                        shutil.copy(community_input_path, community_output_csv)
            except Exception as e:
                self.log(f"[ERROR] Failed to check community column: {str(e)}")
                return

            if not already_has_community:
                if not self.run_cmd([
                    "community",
                    "--out-csv", community_output_csv,
                    "--out-graph", graph_output_gml,
                    community_input_path,
                    "--algo", config['algo'],
                    "-k", str(int(config['k']))
                ], "Component 2: Community detection"):
                    return
            else:
                graph_output_gml = None

            # Merge
            merged_output_csv = os.path.join(config['workdir'], "absa_community.csv")
            if not self.run_cmd([
                "absa_community_merge",
                "--aspect", absa_output_csv,
                "--meta", community_output_csv,
                "--output", merged_output_csv
            ], "Component 3: ABSA + Community Merge"):
                return

            # Consensus
            consensus_output_txt = os.path.join(config['workdir'], "consensus.txt")
            consensus_details_txt = os.path.join(config['workdir'], "consensus_details.txt")
            if not self.run_cmd([
                "consensus",
                "-o", consensus_output_txt,
                "--details", consensus_details_txt,
                merged_output_csv
            ], "Component 4: Consensus"):
                return

            total_elapsed = time.time() - total_start
            self.log("\n[PIPELINE COMPLETE]")
            self.log(f"Final consensus file: {consensus_output_txt}")
            self.log("\n[PIPELINE TIMINGS]")
            for comp, t in self.timings.items():
                self.log(f"{comp}: {t:.2f}s")
            self.log(f"Total Time: {total_elapsed:.2f}s")

        finally:
            self.is_running = False



def create_gui():
    pipeline = PipelineGUI()
    config = {
        'workdir': 'pipeline_out',
        'model': 'gemini',
        'model_name': 'gemini-2.0-flash',
        'batch_size': 25,
        'algo': 'louvain',
        'k': 2,
        'vector_mode': 'tfidf'
    }

    # Model options
    gemini_models = ['gemini-2.0-flash', 'gemini-2.5-flash']
    dp_processors = ['stanza', 'udpipe']
    vector_modes = ['sentence', 'tfidf', 'fasttext']

    @ui.page('/')
    def main_page():
        with ui.column().classes('w-[880px] mx-auto space-y-6'):
            # Header
            ui.markdown('# Echo-ABSA-GDM Pipeline').classes('text-center mb-8')
            
            # File Upload Section
            with ui.card().classes('w-full p-6'):
                ui.label('File Uploads').classes('text-xl font-semibold mb-4')
                
                with ui.row().classes('w-full gap-4'):
                    with ui.column().classes('flex-1'):
                        ui.label('Community Input CSV *').classes('text-sm font-medium')
                        ui.upload(
                            auto_upload=True, 
                            on_upload=lambda e: pipeline.handle_file_upload(e, 'community_input')
                        ).props('accept=.csv').classes('w-full')
                    
                    with ui.column().classes('flex-1'):
                        ui.label('ABSA Aspect CSV *').classes('text-sm font-medium')
                        ui.upload(
                            auto_upload=True,
                            on_upload=lambda e: pipeline.handle_file_upload(e, 'absa_aspect')
                        ).props('accept=.csv').classes('w-full')

            # General Settings
            with ui.card().classes('w-full p-6'):
                ui.label('General Settings').classes('text-xl font-semibold mb-4')
                
                ui.label('Working Directory').classes('text-sm font-medium')
                workdir_input = ui.input(value=config['workdir']).classes('w-full max-w-md')
                workdir_input.bind_value(config, 'workdir')

            # Model Configuration
            with ui.card().classes('w-full p-6'):
                ui.label('Model Configuration').classes('text-xl font-semibold mb-4')
                
                with ui.row().classes('w-full gap-4'):
                    with ui.column().classes('flex-1'):
                        ui.label('Model Type *').classes('text-sm font-medium')
                        model_select = ui.select(['gemini', 'dp'], value=config['model']).classes('w-full')
                        model_select.bind_value(config, 'model')
                    
                    with ui.column().classes('flex-1'):
                        # Conditional model name dropdown
                        model_name_container = ui.column().classes('w-full')
                        
                        def update_model_options():
                            model_name_container.clear()
                            with model_name_container:
                                ui.label('Model/Processor *').classes('text-sm font-medium')
                                if config['model'] == 'gemini':
                                    model_dropdown = ui.select(gemini_models, value=config['model_name']).classes('w-full')
                                    model_dropdown.bind_value(config, 'model_name')
                                else:
                                    processor_dropdown = ui.select(dp_processors, value='stanza').classes('w-full')
                                    processor_dropdown.bind_value(config, 'model_name')
                        
                        model_select.on('update:model-value', lambda: update_model_options())
                        update_model_options()

            # Conditional Settings
            conditional_container = ui.column().classes('w-full space-y-6')
            
            def update_conditional_settings():
                conditional_container.clear()
                with conditional_container:
                    if config['model'] == 'gemini':
                        with ui.card().classes('w-full p-6'):
                            ui.label('Gemini Settings').classes('text-xl font-semibold mb-4')
                            ui.label('Batch Size').classes('text-sm font-medium')
                            batch_input = ui.number(value=config['batch_size'], min=1, max=100).classes('w-full max-w-xs')
                            batch_input.bind_value(config, 'batch_size')
                    
                    elif config['model'] == 'dp':
                        with ui.card().classes('w-full p-6'):
                            ui.label('DP Settings').classes('text-xl font-semibold mb-4')
                            
                            with ui.row().classes('w-full gap-4'):
                                with ui.column().classes('flex-1'):
                                    ui.label('Vector Mode').classes('text-sm font-medium')
                                    vector_select = ui.select(vector_modes, value=config['vector_mode']).classes('w-full')
                                    vector_select.bind_value(config, 'vector_mode')
                                
                                with ui.column().classes('flex-1'):
                                    ui.label('Lexicon CSV *').classes('text-sm font-medium')
                                    ui.upload(
                                        auto_upload=True,
                                        on_upload=lambda e: pipeline.handle_file_upload(e, 'lexicon_file')
                                    ).props('accept=.csv').classes('w-full')

            model_select.on('update:model-value', lambda: update_conditional_settings())
            update_conditional_settings()

            # Community Detection
            with ui.card().classes('w-full p-6'):
                ui.label('Community Detection').classes('text-xl font-semibold mb-4')
                
                with ui.row().classes('w-full gap-4'):
                    with ui.column().classes('flex-1'):
                        ui.label('Algorithm').classes('text-sm font-medium')
                        algo_select = ui.select(['louvain', 'metis'], value=config['algo']).classes('w-full')
                        algo_select.bind_value(config, 'algo')
                    
                    with ui.column().classes('flex-1'):
                        # Conditional k parameter
                        k_container = ui.column().classes('w-full')
                        
                        def update_k_visibility():
                            k_container.clear()
                            if config['algo'] == 'metis':
                                with k_container:
                                    ui.label('Number of Communities').classes('text-sm font-medium')
                                    k_input = ui.number(value=config['k'], min=2).classes('w-full')
                                    k_input.bind_value(config, 'k')
                        
                        algo_select.on('update:model-value', lambda: update_k_visibility())
                        update_k_visibility()

            # Run Button Section
            with ui.card().classes('w-full p-6 text-center'):
                run_button = ui.button('Run Pipeline', color='primary').classes('text-lg px-12 py-4')
                
                def validate_and_run():
                    if pipeline.is_running:
                        ui.notify('Pipeline already running!', type='warning')
                        return
                    
                    # Validation
                    if 'community_input' not in pipeline.uploaded_files:
                        ui.notify('Please upload Community Input CSV', type='negative')
                        return
                    if 'absa_aspect' not in pipeline.uploaded_files:
                        ui.notify('Please upload ABSA Aspect CSV', type='negative')
                        return
                    if config['model'] == 'dp' and 'lexicon_file' not in pipeline.uploaded_files:
                        ui.notify('Please upload Lexicon CSV for DP model', type='negative')
                        return
                    
                    # Clear logs and start
                    log_area.set_value('')
                    ui.notify('Pipeline started!', type='positive')
                    threading.Thread(target=pipeline.run_pipeline, args=(config.copy(),), daemon=True).start()
                
                run_button.on('click', validate_and_run)

            # Status Section
            with ui.card().classes('w-full p-6'):
                ui.label('Status').classes('text-xl font-semibold mb-4')
                status_container = ui.column().classes('w-full')

            # Logs Section
            with ui.card().classes('w-full p-6'):
                ui.label('Pipeline Logs').classes('text-xl font-semibold mb-4')
                log_area = ui.textarea().classes('w-full h-96 font-mono text-sm').props('readonly outlined')

        # Update functions
        def update_logs():
            content = ''
            while not pipeline.log_queue.empty():
                content += pipeline.log_queue.get() + '\n'
            if content:
                current_value = log_area.value if log_area.value else ''
                log_area.set_value(current_value + content)
                
        def update_status():
            status_container.clear()
            with status_container:
                # Check file uploads status
                community_uploaded = 'community_input' in pipeline.uploaded_files
                aspect_uploaded = 'absa_aspect' in pipeline.uploaded_files
                lexicon_uploaded = 'lexicon_file' in pipeline.uploaded_files
                lexicon_required = config['model'] == 'dp'
                
                if pipeline.is_running:
                    ui.label('Running Pipeline...').classes('text-lg text-blue-600 font-medium')
                    run_button.props('loading')
                elif not community_uploaded or not aspect_uploaded or (lexicon_required and not lexicon_uploaded):
                    ui.label('Waiting for required files').classes('text-lg text-orange-600 font-medium')
                    
                    # Show missing files
                    missing_files = []
                    if not community_uploaded:
                        missing_files.append('Community Input CSV')
                    if not aspect_uploaded:
                        missing_files.append('ABSA Aspect CSV')
                    if lexicon_required and not lexicon_uploaded:
                        missing_files.append('Lexicon CSV')
                    
                    ui.label(f'Missing: {", ".join(missing_files)}').classes('text-sm text-gray-600 mt-1')
                    run_button.props(remove='loading')
                else:
                    ui.label('Ready to run').classes('text-lg text-green-600 font-medium')
                    
                    # Show uploaded files
                    uploaded_files = []
                    if community_uploaded:
                        uploaded_files.append('Community Input CSV')
                    if aspect_uploaded:
                        uploaded_files.append('ABSA Aspect CSV')
                    if lexicon_uploaded:
                        uploaded_files.append('Lexicon CSV')
                    
                    ui.label(f'Files ready: {", ".join(uploaded_files)}').classes('text-sm text-gray-600 mt-1')
                    run_button.props(remove='loading')

        # Timers
        ui.timer(0.5, lambda: [update_logs(), update_status()])

    # Custom CSS for better styling
    ui.add_head_html('''
        <style>
            .nicegui-content {
                padding: 2rem;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
            }
            .q-card {
                border-radius: 12px;
                box-shadow: 0 8px 32px rgba(0,0,0,0.1);
                backdrop-filter: blur(10px);
                background: rgba(255,255,255,0.95);
            }
            .q-btn {
                border-radius: 8px;
                font-weight: 600;
            }
            .q-input, .q-select {
                border-radius: 6px;
            }
        </style>
    ''')

    ui.run(title='Echo-ABSA-GDM Pipeline', host='0.0.0.0', port=8080, dark=False)

if __name__ in {"__main__", "__mp_main__"}:
    create_gui()