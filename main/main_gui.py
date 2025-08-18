import csv
import subprocess
import os
import sys
import glob
import asyncio
from pathlib import Path
from typing import Optional
from nicegui import ui, app, events
import threading
import queue
import time

class PipelineGUI:
    def __init__(self):
        self.log_queue = queue.Queue()
        self.current_process = None
        self.is_running = False
        
    def run_cmd(self, cmd, description):
        """Run command and capture output"""
        self.log(f"\n[INFO] Running: {description}")
        self.log(f"[CMD] {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if result.returncode != 0:
                self.log(f"[ERROR] {description} failed:\n{result.stderr}")
                return False
            else:
                self.log(f"[DONE] {description} complete.")
                if result.stdout.strip():
                    self.log(result.stdout.strip())
                return True
        except Exception as e:
            self.log(f"[ERROR] Failed to run {description}: {str(e)}")
            return False

    def log(self, message):
        """Add message to log queue"""
        self.log_queue.put(message)

    def create_absa_input_from_community(self, community_csv_path, absa_input_path):
        """Create ABSA input CSV from community input"""
        try:
            with open(community_csv_path, newline="", encoding="utf-8") as infile, \
                 open(absa_input_path, "w", newline="", encoding="utf-8") as outfile:

                reader = csv.DictReader(infile)
                fieldnames = ["id", "text"]
                writer = csv.DictWriter(outfile, fieldnames=fieldnames)
                writer.writeheader()

                for row in reader:
                    if "id" not in row or "text" not in row:
                        self.log("[ERROR] community_input CSV must contain 'id' and 'text' columns")
                        return False
                    writer.writerow({"id": row["id"], "text": row["text"]})
            return True
        except Exception as e:
            self.log(f"[ERROR] Failed to create ABSA input: {str(e)}")
            return False

    async def run_pipeline(self, config):
        """Run the complete pipeline with given configuration"""
        self.is_running = True
        
        try:
            # Create working directory
            os.makedirs(config['workdir'], exist_ok=True)

            # Auto-create absa_input.csv
            absa_input_csv = os.path.join(config['workdir'], "absa_input.csv")
            if not self.create_absa_input_from_community(config['community_input'], absa_input_csv):
                return

            # Step 1: ABSA model execution
            aspect_output_folder = os.path.join(config['workdir'], "aspect_out")
            
            if config['model'] == "gemini":
                model_value = config.get('model_name', 'gemini-2.5-pro')
                success = self.run_cmd([
                    "gemini",
                    "--input", absa_input_csv,
                    "--aspects", config['absa_aspect'],
                    "--output", aspect_output_folder,
                    "--model", model_value,
                    "--mode", "block",
                    "--batch_size", str(config['batch_size']),
                    "--conf_thresholds", str(config['conf_threshold'])
                ], "Component 1: Gemini")
                
            elif config['model'] == "dp":
                success = self.run_cmd([
                    "dp",
                    "--input", absa_input_csv,
                    "--aspects", config['absa_aspect'],
                    "--lexicon", "lexicon.csv",
                    "--output", aspect_output_folder,
                    "--vector_mode", "tfidf",
                    "--max_iter", "50"
                ], "Component 1: Double Propagation (DP)")
                
            elif config['model'] == "gpt":
                model_value = config.get('model_name', 'gpt-4.1')
                cmd = [
                    "gptabsa",
                    "--input", absa_input_csv,
                    "--aspects", config['absa_aspect'],
                    "--output", aspect_output_folder,
                    "--mode", config['gpt_mode'],
                    "--batch_size", str(config['batch_size']),
                    "--model", model_value,
                    "--conf_thresholds", str(config['conf_threshold'])

                ]
                success = self.run_cmd(cmd, "Component 1: GPT ABSA")

            if not success:
                return

            # Find ABSA output CSV
            if config['model'] == "gemini":
                pattern = os.path.join(aspect_output_folder,
                                     f"absa_input_{int(config['conf_threshold'] * 10):02d}_*.csv")
            else:
                pattern = os.path.join(aspect_output_folder, "*.csv")

            matching_files = glob.glob(pattern)
            absa_output_csv = matching_files[0] if matching_files else None

            if not absa_output_csv:
                self.log("[ERROR] No ABSA output file found")
                return

            # Step 2: Community detection
            community_output_csv = os.path.join(config['workdir'], "community_out.csv")
            edges_output_csv = community_output_csv.replace('.csv', '_edges.csv')
            graph_output_gml = os.path.join(config['workdir'], "community_graph.gml")

            success = self.run_cmd([
                "community",
                "--out-csv", community_output_csv,
                "--out-graph", graph_output_gml,
                config['community_input'],
                "--algo", config['algo'],
                "-k", str(config['k']),
            ], "Component 2: Community detection (nodes)")

            if not success:
                return

            # Step 3: ABSA + community merge
            merged_output_csv = os.path.join(config['workdir'], "absa_community.csv")
            success = self.run_cmd([
                "absa_community_merge",
                "--aspect", absa_output_csv,
                "--meta", community_output_csv,
                "--output", merged_output_csv,
            ], "Component 3: ABSA + Community Merge")

            if not success:
                return

            # Step 4: Consensus
            consensus_output_txt = os.path.join(config['workdir'], "consensus.txt")
            consensus_details_txt = os.path.join(config['workdir'], "consensus_details.txt")
            success = self.run_cmd([
                "consensus",
                "-p", str(config['consensus_p']),
                "-o", consensus_output_txt,
                "--details", consensus_details_txt,
                edges_output_csv, 
                merged_output_csv
            ], "Component 4: Consensus")

            if success:
                self.log("\n[PIPELINE COMPLETE]")
                self.log(f"Final consensus file: {consensus_output_txt}")
            
        except Exception as e:
            self.log(f"[ERROR] Pipeline failed: {str(e)}")
        finally:
            self.is_running = False

def create_gui():
    pipeline = PipelineGUI()
    
    # Configuration state
    config = {
        'community_input': '',
        'absa_aspect': '',
        'workdir': 'pipeline_out',
        'model': 'gemini',
        'model_name': 'gemini-1.5-pro',
        'batch_size': 30,
        'consensus_p': 0.5,
        'conf_threshold': 0.6,
        'conf_thresholds': '',
        'few_shot': '',
        'gpt_mode': 'block',
        'algo': 'louvain',
        'k': 2,
        'lexicon_file': ''
    }

    @ui.page('/')
    def main_page():
        ui.page_title('Echo-ABSA-GDM')
        
        with ui.column().classes('w-full max-w-4xl mx-auto p-4'):
            ui.markdown('# Echo-ABSA-GDM')
            ui.markdown('Configure and run the echo chamber assessment pipeline using ABSA, community detection, and GDM.')
            
            # File inputs section
            with ui.card().classes('w-full p-4'):
                ui.label('Input Files').classes('text-lg font-semibold')
                
                with ui.row().classes('w-full gap-4'):
                    with ui.column().classes('flex-1'):
                        ui.label('Community Input CSV*')
                        community_upload = ui.upload(
                            on_upload=lambda e: setattr(config, 'community_input', e.name),
                            auto_upload=True
                        ).classes('w-full').props('accept=.csv')
                        ui.label().bind_text_from(config, 'community_input', 
                                                lambda x: f'Selected: {x}' if x else 'No file selected').classes('text-sm text-gray-600')
                        
                    with ui.column().classes('flex-1'):
                        ui.label('ABSA Aspect CSV*')
                        aspect_upload = ui.upload(
                            on_upload=lambda e: setattr(config, 'absa_aspect', e.name),
                            auto_upload=True
                        ).classes('w-full').props('accept=.csv')
                        ui.label().bind_text_from(config, 'absa_aspect', 
                                                lambda x: f'Selected: {x}' if x else 'No file selected').classes('text-sm text-gray-600')
                
                ui.label('Working Directory')
                workdir_input = ui.input(
                    placeholder='pipeline_out',
                    value=config['workdir']
                ).classes('w-full')
                workdir_input.bind_value(config, 'workdir')

            # Model configuration section
            with ui.card().classes('w-full p-4'):
                ui.label('Model Configuration').classes('text-lg font-semibold')
                
                # Model type and name dropdowns
                model_options = {
                    'gemini': ['gemini-1.5-pro', 'gemini-1.5-flash', 'gemini-2.0-flash-exp', 'gemini-2.5-pro'],
                    'gpt': ['gpt-4o', 'gpt-4o-mini', 'gpt-4-turbo', 'gpt-3.5-turbo'],
                    'dp': ['N/A (Dictionary-based)']
                }
                
                with ui.row().classes('w-full gap-4'):
                    with ui.column().classes('flex-1'):
                        ui.label('Model Type')
                        model_select = ui.select(
                            ['gemini', 'dp', 'gpt'],
                            value=config['model']
                        ).classes('w-full')
                        model_select.bind_value(config, 'model')
                        
                    with ui.column().classes('flex-1'):
                        ui.label('Model Name')
                        model_name_select = ui.select(
                            model_options[config['model']],
                            value=config['model_name']
                        ).classes('w-full')
                        
                        # Handle model name selection manually
                        def on_model_name_change(e):
                            config['model_name'] = e.value
                        
                        model_name_select.on('update:model-value', on_model_name_change)
                
                with ui.row().classes('w-full gap-4'):
                    with ui.column().classes('flex-1'):
                        ui.label('Batch Size')
                        batch_size_input = ui.number(
                            value=config['batch_size'],
                            min=1,
                            max=100
                        ).classes('w-full')
                        batch_size_input.bind_value(config, 'batch_size')
                        
                    with ui.column().classes('flex-1'):
                        ui.label('Consensus Threshold P')
                        consensus_input = ui.number(
                            value=config['consensus_p'],
                            min=0.0,
                            max=1.0,
                            step=0.1
                        ).classes('w-full')
                        consensus_input.bind_value(config, 'consensus_p')
                
                # Update model name options when model type changes
                def update_model_names():
                    options = model_options[config['model']]
                    model_name_select.options = options
                    config['model_name'] = options[0]
                    model_name_select.value = options[0]
                    model_name_select.update()
                
                model_select.on('update:model-value', lambda: update_model_names())

            # Model-specific options
            with ui.card().classes('w-full p-4'):
                ui.label('Model-Specific Options').classes('text-lg font-semibold')
                
                # Confidence threshold (for Gemini and GPT)
                conf_threshold_container = ui.row().classes('w-full gap-4')
                with conf_threshold_container:
                    with ui.column().classes('flex-1'):
                        ui.label('Confidence Threshold')
                        conf_threshold_input = ui.number(
                            value=config['conf_threshold'],
                            min=0.0,
                            max=1.0,
                            step=0.1
                        ).classes('w-full')
                        conf_threshold_input.bind_value(config, 'conf_threshold')
                
                # DP-specific options
                dp_container = ui.column().classes('w-full')
                with dp_container:
                    ui.label('Double Propagation Options').classes('font-medium')
                    ui.label('Lexicon CSV File')
                    lexicon_upload = ui.upload(
                        on_upload=lambda e: setattr(config, 'lexicon_file', e.name),
                        auto_upload=True
                    ).classes('w-full').props('accept=.csv')
                    ui.label().bind_text_from(config, 'lexicon_file', 
                                            lambda x: f'Selected: {x}' if x else 'No file selected (will use default lexicon.csv)').classes('text-sm text-gray-600')
                
                # Update visibility based on model selection
                def update_model_options():
                    # Show confidence threshold for Gemini and GPT only
                    conf_threshold_container.set_visibility(config['model'] in ['gemini', 'gpt'])
                    # Show DP options only for DP model
                    dp_container.set_visibility(config['model'] == 'dp')
                    
                model_select.on('update:model-value', lambda: update_model_options())
                update_model_options()

            # Community detection options
            with ui.card().classes('w-full p-4'):
                ui.label('Community Detection').classes('text-lg font-semibold')
                
                with ui.row().classes('w-full gap-4'):
                    with ui.column().classes('flex-1'):
                        ui.label('Algorithm')
                        algo_select = ui.select(
                            ['louvain', 'metis'],
                            value=config['algo']
                        ).classes('w-full')
                        algo_select.bind_value(config, 'algo')
                        
                    # Always create the k_input container but control visibility
                    k_container = ui.column().classes('flex-1')
                    with k_container:
                        ui.label('Number of Communities (METIS only)')
                        k_input = ui.number(
                            value=config['k'],
                            min=2,
                            max=100
                        ).classes('w-full')
                        k_input.bind_value(config, 'k')
                
                # Update visibility based on algorithm selection
                def update_algo_options():
                    k_container.set_visibility(config['algo'] == 'metis')
                
                algo_select.on('update:model-value', lambda: update_algo_options())
                update_algo_options()  # Set initial visibility

            # Control buttons
            with ui.card().classes('w-full p-4'):
                with ui.row().classes('w-full gap-4'):
                    async def run_pipeline_click():
                        # Validate files first
                        errors = []
                        if not config['community_input']:
                            errors.append('Community input file is required')
                        if not config['absa_aspect']:
                            errors.append('ABSA aspect file is required')
                        if config['model'] == 'dp' and not config.get('lexicon_file'):
                            errors.append('Lexicon file is required for DP model (or ensure lexicon.csv exists in working directory)')
                        
                        if errors:
                            ui.notify('\n'.join(errors), type='negative')
                            return
                        
                        if pipeline.is_running:
                            ui.notify('Pipeline is already running', type='warning')
                            return
                        
                        # All validations passed
                        ui.notify('Files validated successfully', type='positive')
                        log_area.set_value('')  # Clear previous logs
                        ui.notify('Pipeline started', type='positive')
                        
                        # Run pipeline in background thread
                        def run_in_thread():
                            asyncio.run(pipeline.run_pipeline(config.copy()))
                        
                        threading.Thread(target=run_in_thread, daemon=True).start()
                    
                    run_button = ui.button('Run Pipeline', on_click=run_pipeline_click).classes('bg-green-500')

            # Log output section
            with ui.card().classes('w-full p-4'):
                ui.label('Pipeline Output').classes('text-lg font-semibold')
                log_area = ui.textarea().classes('w-full h-64 font-mono text-sm').props('readonly')
                
                # Update log area periodically
                log_timer = None
                
                async def update_logs():
                    nonlocal log_timer
                    content = log_area.value or ''
                    try:
                        while True:
                            message = pipeline.log_queue.get_nowait()
                            content += message + '\n'
                    except queue.Empty:
                        pass
                    
                    if content != log_area.value:
                        log_area.set_value(content)
                        # Auto-scroll to bottom
                        try:
                            await ui.run_javascript('''
                                const textarea = document.querySelector('textarea');
                                if (textarea) textarea.scrollTop = textarea.scrollHeight;
                            ''')
                        except:
                            pass  # Ignore if client disconnected
                
                # Start timer only when page is active
                def start_log_timer():
                    nonlocal log_timer
                    if log_timer is None:
                        log_timer = ui.timer(0.5, update_logs)
                
                start_log_timer()

    ui.run(title='ABSA Pipeline Orchestrator', dark=False, host='0.0.0.0', port=8080)

if __name__ in {"__main__", "__mp_main__"}:
    create_gui()