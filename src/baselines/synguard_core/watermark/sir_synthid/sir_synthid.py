from __future__ import annotations
from watermark.sir.sir import SIR, SIRLogitsProcessor, SIRConfig, SIRUtils
from watermark.synthid.synthid import SynthID, SynthIDLogitsProcessor, SynthIDConfig, SynthIDUtils

from watermark.base import BaseWatermark, BaseConfig
from functools import partial
import jieba
from transformers import LogitsProcessorList
from utils.transformers_config import TransformersConfig

from visualize.data_for_visualization import DataForVisualization
import json
from exceptions.exceptions import AlgorithmNameMismatchError, InvalidWatermarkModeError


# yese fhsfifeogkgnjgfgjk/glrgl
class SIR_SynthID_Utils:

    def load_config_file(path: str) -> dict:
      """Load a JSON configuration file from the specified path and return it as a dictionary."""
      try:
          with open(path, 'r') as f:
              config_dict = json.load(f)
          return config_dict

      except FileNotFoundError:
          print(f"Error: The file '{path}' does not exist.")
          return None
      except json.JSONDecodeError as e:
          print(f"Error decoding JSON in '{path}': {e}")
          # Handle other potential JSON decoding errors here
          return None
      except Exception as e:
          print(f"An unexpected error occurred: {e}")
          # Handle other unexpected errors here
          return None
      
class MySIRConfig(BaseConfig):
    """Config class for SIR algorithm, load config file and initialize parameters."""
  
    def initialize_parameters(self) -> None:
        """Initialize algorithm-specific parameters."""
        self.delta = self.config_dict['delta']
        self.chunk_length = self.config_dict['chunk_length']
        self.scale_dimension = self.config_dict['scale_dimension']
        self.z_threshold = self.config_dict['z_threshold']
        self.transform_model_input_dim = self.config_dict['transform_model_input_dim']
        self.transform_model_name = self.config_dict['transform_model_name']
        self.embedding_model_path = self.config_dict['embedding_model_path']
        self.mapping_name = self.config_dict['mapping_name']
  
    @property
    def algorithm_name(self) -> str:
        """Return the algorithm name."""
        return 'SIR_SynthID'
    
class MySynthIDConfig(BaseConfig):
    """Config class for SynthID algorithm, load config file and initialize parameters."""
    
    def initialize_parameters(self) -> None:
        """Initialize algorithm-specific parameters."""
        self.ngram_len = self.config_dict['ngram_len']
        self.keys = self.config_dict['keys']
        self.sampling_table_size = self.config_dict['sampling_table_size']
        self.sampling_table_seed = self.config_dict['sampling_table_seed']
        self.context_history_size = self.config_dict['context_history_size']
        self.detector_name = self.config_dict['detector_type']
        self.threshold = self.config_dict['threshold']
        self.watermark_mode = self.config_dict['watermark_mode']
        self.num_leaves = self.config_dict['num_leaves']

        # Validate detect mode
        if self.watermark_mode not in ['distortionary', 'non-distortionary']:
            raise InvalidWatermarkModeError(self.watermark_mode)
        
        self.top_k = getattr(self.transformers_config, 'top_k', -1)
        self.temperature = getattr(self.transformers_config, 'temperature', 0.7)
  
        
    @property
    def algorithm_name(self) -> str:
        """Return the algorithm name."""
        return 'SIR_SynthID'

class SIR_SynthID_Config(BaseConfig):
    """Combined configuration for both SIR and SynthID algorithms."""

    def __init__(self, 
                 algorithm_config_path: str,
                 transformers_config: TransformersConfig,
                 *args, **kwargs) -> None:
        """
        Initialize combined config with both SIR and SynthID configs.

        Parameters:
            algorithm_config_path (str): Path to config file.
            transformers_config (TransformersConfig): Huggingface transformer config.
            model_delta (float): Custom parameter for this combined config.
        """
        # Initialize parent (BaseConfig)
        super().__init__(algorithm_config_path, transformers_config, *args, **kwargs)
        # 从 config_dict 中读取自定义参数
        self.model_delta = self.config_dict.get("model_delta")  # 默认值为 0.5

        # Create sub-config instances
        self.sir_config = SIRConfig(algorithm_config_path, transformers_config, *args, **kwargs)
        self.synthid_config = SynthIDConfig(algorithm_config_path, transformers_config, *args, **kwargs)

    def initialize_parameters(self) -> None:
        """Optional: No algorithm-specific parameters for the combined config itself."""
        # This method can be left empty or used to unify settings from both configs if needed.
        pass

    @property
    def algorithm_name(self) -> str:
        return "sir_synthid"

    
# class SIR_SynthID_Config(BaseConfig):
#     """Combined config for both SIR and SynthID algorithms."""
#     def __init__(self, algorithm_config, transformers_config):
#         super().__init__(algorithm_config, transformers_config)
#         self.sir_config = SIRConfig(algorithm_config, transformers_config)
#         self.synthid_config = SynthIDConfig(algorithm_config, transformers_config)

#         # Initialize algorithm-specific parameters
#         self.initialize_parameters()
    
#     def initialize_parameters(self) -> None:
#         """Initialize both SIR and SynthID parameters."""
#         # self.sir_config.initialize_parameters()
#         # self.synthid_config.initialize_parameters()
#         self.model_delta = self.config_dict['model_delta']

#     @property
#     def algorithm_name(self) -> str:
        """Return the algorithm name."""
        return 'SIR_SynthID'

class SIR_SynthID(BaseWatermark):
    """Top-level class for SIR_SynthID algorithm."""

    # 这里传入的参数 algorithm_config 是 SIR_SynthID_Config 对象
    def __init__(self, algorithm_config: str | SIR_SynthID_Config, transformers_config: TransformersConfig | None = None, *args, **kwargs) -> None:
        """
            Initialize the SIR algorithm.  

            Parameters:
                algorithm_config (str): Path to the algorithm configuration file.
                transformers_config (TransformersConfig): Configuration for the transformers model.
        """
        # if isinstance(algorithm_config, str):
        #     self.config_sir = MySIRConfig(algorithm_config, transformers_config)
        #     self.config_synthid = MySynthIDConfig(algorithm_config, transformers_config)
        # else:
        #     raise TypeError("algorithm_config must be a path string ")
   
        if isinstance(algorithm_config, str):
            self.config = SIR_SynthID_Config(algorithm_config, transformers_config)
        elif isinstance(algorithm_config, SIR_SynthID_Config):
            self.config = algorithm_config
        else:
            raise TypeError("algorithm_config must be either a path string or a SIR_SynthID_Config instance")
        
       
        self.utils_sir = SIRUtils(self.config.sir_config)
        self.utils_synthid = SynthIDUtils(self.config.synthid_config)
        self.logits_processor_sir = SIRLogitsProcessor(self.config.sir_config, self.utils_sir)
        self.logits_processor_synthid = SynthIDLogitsProcessor(self.config.synthid_config, self.utils_synthid)


        # 这里传入的参数 algorithm_config 是 SIR_SynthID_Config 对象
        # 已更新为 直接使用 SIRConfig 和 SynthIDConfig 对象
        self.watermark_sir = SIR(self.config.sir_config, transformers_config, *args, **kwargs)
        self.watermark_synthid = SynthID(self.config.synthid_config, transformers_config, *args, **kwargs)

    def generate_watermarked_text(self, prompt: str, *args, **kwargs):
        """Generate watermarked text."""

        # Configure generate_with_watermark
        generate_with_watermark = partial(
            self.config.generation_model.generate,
            logits_processor=LogitsProcessorList([self.logits_processor_sir, self.logits_processor_synthid]), 
            **self.config.gen_kwargs
        )
        
        # encode prompt
        encoded_prompt = self.config.generation_tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to(self.config.device)
        # generate watermarked text
        encoded_watermarked_text = generate_with_watermark(**encoded_prompt)
        # decode
        watermarked_text = self.config.generation_tokenizer.batch_decode(encoded_watermarked_text, skip_special_tokens=True)[0]
        return watermarked_text

    '''改变 delta 的值 (0.2 -> 0.8), 画出不同值下的 AOC 曲线 TODO''' 
    def detect_watermark(self, text: str, model_delta: float = 0.5, zscore_threshold : float = 0.5, return_dict: bool = True, *args, **kwargs):
      """Detect watermark using combined SIR + SynthID score."""
      '''
      Args:
        delta:
          delta = 0.0 → 完全使用 SIR 检测
          delta = 0.5 → 平均融合
          delta = 1.0 → 完全使用 SynthID 检测

        zscore_threshold:
          default value is 0.5
          can get via making False Positive Possibility as 0.1% or 0.5% ??
          or just by (1- delta) * sit_threshold + delta * synthid_threshold

      '''
      model_delta = self.config.model_delta

      sir_score = ( self.watermark_sir.detect_watermark(text) )['score']     # ∈ [-1, 1]
      synthid_score = ( self.watermark_synthid.detect_watermark(text) )['score']  # ∈ [0, 1]

      # Normalize sir_score to [0, 1]
      sir_score_norm = (float(sir_score) + 1) / 2
      sir_threshold = (self.config.sir_config.z_threshold + 1) / 2

      synthid_threshold = self.config.synthid_config.threshold

      # Weighted combination
      combined_score = (1 - model_delta) * sir_score_norm + model_delta * float(synthid_score)

      # Threshold could be configurable or set empirically
      zscore_threshold = (1 - model_delta) * sir_threshold + model_delta * synthid_threshold
      is_watermarked = combined_score > zscore_threshold

      if return_dict:
          return {
              "is_watermarked": is_watermarked,
              "score": combined_score,
              "sir_score": sir_score,
              "synthid_score": synthid_score,
              "model_delta": model_delta
          }
      else:
          return (is_watermarked, combined_score)







    '''原始的 SIR 的detection 函数'''
    # def detect_watermark(self, text: str, return_dict: bool = True, *args, **kwargs):
    #     """Detect watermark in the input text."""

    #     # Split the text into a 2D array of words
    #     word_2d = self.utils.get_text_split(text)

    #     # Initialize a list to hold all computed values for similarity
    #     all_value = []

    #     # Iterate over each sentence in the split text, skipping the first
    #     for i in range(1, len(word_2d)):
    #         # Create the context sentence from all previous text portions
    #         context_sentence = ''.join([''.join(group) for group in word_2d[:i]]).strip()
    #         # Current sentence to check against the context
    #         current_sentence = ''.join(word_2d[i]).strip()

    #         # Continue if the context sentence is shorter than the required chunk length
    #         if len(list(jieba.cut(context_sentence))) < self.config.chunk_length:
    #             continue

    #         # Get embedding of the context sentence
    #         context_embedding = self.utils.get_embedding(context_sentence)
    #         # Transform the embedding using the model, process output
    #         output = self.utils.transform_model(context_embedding).cpu().detach()[0].numpy()
    #         # Scale the output vector and map to predefined indices
    #         similarity_array = self.utils.scale_vector(output)[self.utils.mapping]

    #         # Encode the current sentence into tokens
    #         tokens = self.config.generation_tokenizer.encode(current_sentence, return_tensors="pt", add_special_tokens=False)

    #         # Append negative similarity values for each token in the current sentence
    #         for index in tokens[0]:
    #             all_value.append(-float(similarity_array[index]))

    #     # Calculate the mean of all similarity values
    #     z_score = np.mean(all_value)

    #     # Determine if the z_score indicates a watermark
    #     is_watermarked = z_score > self.config.z_threshold

    #     # Return results based on the return_dict flag
    #     if return_dict:
    #         return {"is_watermarked": is_watermarked, "score": z_score}
    #     else:
    #         return (is_watermarked, z_score)
        
        # 新函数 暂不提供 数据可视化
    # def get_data_for_visualization(self, text: str, *args, **kwargs):
    #     """Get data for visualization."""
        
    #     # Split the text into 2D array of words
    #     word_2d = self.utils.get_text_split(text)
    #     highlight_values = []
    #     decoded_tokens = []

    #     # Iterate over each sentence in the text
    #     for i in range(len(word_2d)):
    #         # Construct the context sentence from the previous sentences
    #         context_sentence = ' '.join([' '.join(group) for group in word_2d[:i]])
    #         # Current sentence for tokenization
    #         current_sentence = ' '.join(word_2d[i])
    #         # Tokenize the current sentence
    #         tokens = self.config.generation_tokenizer.encode(current_sentence, return_tensors="pt", add_special_tokens=False)

    #         # Decode each token and append to the decoded_tokens list
    #         for token_id in tokens[0]:
    #             token = self.config.generation_tokenizer.decode(token_id.item())
    #             decoded_tokens.append(token)

    #         # If the context sentence is shorter than required, append highlight -1 for each token
    #         if len(context_sentence.split()) < self.config.chunk_length:
    #             highlight_values.extend([-1] * len(tokens[0]))
    #             continue

    #         # Get the embedding of the context sentence and process it through the model
    #         context_embedding = self.utils.get_embedding(context_sentence)
    #         output = self.utils.transform_model(context_embedding).cpu().detach()[0].numpy()

    #         # Scale the output vector and get similarity values
    #         similarity_array = self.utils.scale_vector(output)[self.utils.mapping]

    #         # Append highlight values based on similarity
    #         for token_index in tokens[0]:
    #             similarity_value = -float(similarity_array[token_index.item()])
    #             highlight_values.append(1 if similarity_value > 0 else 0)

    #     return DataForVisualization(decoded_tokens, highlight_values)