### Microsoft's Aurora in JAX

A step-by-step guide to translating Aurora's architecture, crafting a JAX training loop, and benchmarking results.

### Abstract

In May 2025, Microsoft's Aurora paper was published in Nature as "A Foundation Model of the Atmosphere," introducing a 1.3 billion-parameter forecasting model. Although the paper detailed the architecture, the authors did not release end-to-end training code or optimal GPU acceleration strategies. Here, for the first time, we bridge that gap by making public the full training configuration and complete training code for every component described in the original paper. In our implementation, the most challenging aspect was handling the huge model size and vast volume of data, which consumed significant memory, and writing the code in an optimized manner was of utmost importance. To demonstrate this, we used AuroraSmall, a lighter variant provided by the authors and optimized for experimental efficiency, translated its architecture into JAX, implemented a comprehensive training loop, and benchmarked model performance. While AuroraSmall significantly reduces resource demands, its forward-pass times still trail the original PyTorch implementation, highlighting opportunities for further optimization with advanced JAX features. All code and training scripts are now publicly available for the community to use and extend. This article focuses on the code implementation and performance comparison between JAX and PyTorch, rather than revisiting the model details or theoretical rationale already covered by the authors and prior studies.

### What is Aurora?

Aurora is a foundation model developed to significantly enhance Earth system forecasting by leveraging advanced artificial intelligence capabilities. Rather than relying on separate predictive models for various environmental phenomena, Aurora provides a unified forecasting system that can efficiently adapt and perform across multiple domains such as weather, air pollution, ocean waves, and tropical cyclone tracking. Its versatility and ease of fine-tuning offer substantial advantages over traditional forecasting methods, as Aurora matches or surpasses existing numerical and AI-based methods with greater efficiency and accuracy.

Aurora's novelty lies in its foundation model approach, trained on over one million hours of diverse Earth-system data. This extensive pretraining enables Aurora to learn generalized climate dynamics, allowing it to be rapidly fine-tuned for specific forecasting tasks. Its capability to seamlessly address different forecasting scenarios without necessitating multiple specialized models is particularly innovative. Furthermore, Aurora achieves high precision in predicting extreme events such as hurricanes and severe storms, consistently outperforming operational forecasting systems in accuracy and reliability.

Currently, Aurora's open-source codebase, available on GitHub, provides several pretrained model variants tailored specifically for weather, air pollution, and ocean wave forecasts. Users can conveniently install Aurora via pip or conda and quickly deploy accurate forecasts using its user-friendly inference API. The codebase includes comprehensive utility functions such as built-in cyclone tracking via a dedicated Tracker class and extensive support for data handling with NetCDF format integration. Additionally, Aurora offers robust fine-tuning capabilities using Low Rank Adaptation (LoRA), allowing researchers and practitioners to further enhance the model for customized applications. With its efficient inference mechanisms and flexible configuration options, Aurora represents not only a state-of-the-art forecasting tool but also an accessible, practical resource for researchers and environmental specialists aiming to achieve precise and rapid Earth system predictions.

Since its introduction, the Aurora weather model has quickly captured attention, becoming one of the most talked-about AI-driven forecasting tools in recent memory. Enthusiasm for Aurora has surged among both academic researchers and professional meteorologists, reflected by its frequent appearances in major scientific journals and widespread coverage in mainstream media. One key driver behind its swift adoption is its open-source nature, allowing researchers, institutions, and independent developers worldwide to freely access, test, and enhance the model. This spirit of collaboration has cultivated an active community around Aurora, significantly expanding its reach and impact.

Additionally, Aurora's integration into popular platforms like MSN Weather has placed its forecasting capabilities directly into the hands of millions of everyday users, further boosting its visibility and solidifying its reputation as a cutting-edge weather prediction tool. The model has become a central topic at industry conferences and workshops, often highlighted as a benchmark for future advancements in meteorological forecasting. Its increasing popularity can also be seen through the rising number of research articles, citations, and practical applications in the real world, underscoring Aurora's powerful influence on both the scientific community and public perceptions of modern weather prediction technologies.

### JAX vs Pytorch

The choice of deep learning framework is a critical decision in computational research, profoundly influencing not only raw performance but also the clarity, reproducibility, and extensibility of the codebase. Our re-implementation of the Aurora model in Jax, while functionally identical to the original PyTorch version, is motivated by the significant advantages offered by Jax's design philosophy. This section provides a detailed comparison of the two frameworks and outlines the intuitive benefits we expect to realize by this transition.

Jax is fundamentally a library for numerical computing built on a functional programming paradigm. This paradigm encourages the use of "pure" functions --- functions whose output depends solely on their inputs, with no hidden side effects. This contrasts with the more object-oriented and imperative style of PyTorch, where operations can modify tensors in-place. In Jax, all state, such as model parameters or optimizer statistics, must be explicitly passed into a function and its updated version explicitly returned. This explicit handling of state, while seemingly verbose, makes data flow transparent. It eliminates a common class of bugs related to unexpected state changes and significantly enhances the reproducibility and debugability of the model.

The most significant performance advantage of Jax comes from its native integration with the Accelerated Linear Algebra (XLA) compiler. By applying the *@jit* decorator to a Python function, Jax traces the computations, converts them into an intermediate representation, and uses XLA to compile them into highly optimized machine code tailored for the specific accelerator (GPU or TPU) being used. This whole-program, ahead-of-time compilation allows XLA to perform powerful optimizations that are difficult in PyTorch's default eager execution mode. A key optimization is "operator fusion," where multiple individual operations (e.g., a convolution, a bias add, and a ReLU activation) are fused into a single, monolithic kernel. This reduces memory bandwidth requirements and minimizes the overhead of launching multiple separate kernels, often leading to substantial speedups. While PyTorch offers `torch.jit.script` for similar purposes, it is often less seamlessly integrated and may not achieve the same level of aggressive optimization as Jax's JIT-first approach.

Beyond JIT compilation, Jax's power lies in its set of composable function transformations. These transformations --- *grad*, *vmap*, and *pmap* --- allow for expressive and efficient code

**grad**: This function computes the gradient of another function. Because it is a transformation, it can be composed with itself to compute higher-order derivatives, such as Hessians, with ease.

**vmap**: The vectorizing map, *vmap*, is arguably one of Jax's most elegant features. It transforms a function written to operate on a single data point into one that can operate on an entire batch of data, automatically handling the batch dimension. This eliminates the need for manual reshaping and broadcasting logic (e.g., *unsqueeze* or *repeat*), making the code cleaner and more aligned with the underlying mathematical operations on single examples.

**pmap:** The parallel map, *pmap*, is the primary tool for data parallelism. It allows for the execution of the same computation (a "Single Program") on different data across multiple devices ("Multiple Data," or SPMD). This makes scaling up models to run on multiple GPUs or TPUs a conceptually straightforward extension of the core logic.

Despite its advantages, it is important to acknowledge areas where JAX's design choices present trade-offs and where PyTorch may be a more suitable choice. PyTorch boasts a significantly larger and more mature ecosystem, including extensive libraries like `TorchVision` and `TorchAudio`, and a vast number of pre-trained models available through its hub. This makes it exceptionally convenient for rapid prototyping and leveraging existing work. Furthermore, debugging in PyTorch's default eager execution mode is often more straightforward. Errors are raised immediately, and standard Python debugging tools can be used seamlessly. In contrast, debugging JIT-compiled JAX code can be more opaque, as errors may only surface after the compilation stage and can be harder to trace back to the source. For models that rely heavily on dynamic control flow or have complex, data-dependent computational graphs, PyTorch's more imperative style can offer a more natural and less restrictive development experience. Therefore, the choice between JAX and PyTorch is not absolute but depends on the specific priorities of a project, weighing the raw performance and mathematical purity of JAX against the rich ecosystem and developer-friendly debugging of PyTorch.

By translating the Aurora model from PyTorch to Jax, we anticipate a range of concrete benefits. The aggressive optimizations from JIT/XLA are expected to yield significantly faster training iterations and lower inference latency. Finally, the inherent purity and explicit state management of the functional design are projected to create a more robust and maintainable codebase, facilitating future extensions and research on top of the Aurora architecture.

### Architecture of the Model

The Aurora model is a deep learning architecture designed for global weather forecasting. It employs a sophisticated encoder-backbone-decoder structure, processing vast amounts of atmospheric and surface data to predict future weather states. Our starting point is the set of pre-trained weights from the original PyTorch model, which we have converted to the JAX/Flax format to ensure a faithful reproduction of the model's capabilities before fine-tuning. The following diagram and section detail the model's overall structure, its linkage with Low-Rank Adaptation (LoRA) for efficient fine-tuning, and the specific parameter configuration of the smaller variant used in our experiments.

![](https://cdn-images-1.medium.com/max/1440/1*4fe648_PmL_84D6kR6p63w.png)

Aurora Architecture

The model's architecture can be conceptualized in three primary stages:

1\. **Encoder (*Perceiver3DEncoder*)**: The initial stage is responsible for ingesting and transforming the raw input data. It takes distinct inputs for surface-level variables (like temperature and wind speed) and multi-level atmospheric variables (like geopotential and humidity at various pressure levels). Each input variable is embedded into a high-dimensional space. The encoder uses a Perceiver-style architecture with a *PerceiverResampler* to perform cross-attention between a set of learnable latent queries and the embedded atmospheric data. This process effectively distills the information from all atmospheric levels into a compact, fixed-size latent representation. This is then concatenated with the processed surface-level data. Various positional and temporal encodings are added to provide the model with essential spatio-temporal context.

2\. **Backbone (*Swin3DTransformerBackbone*)**: The core of the model is a 3D U-Net-like Swin Transformer. It receives the latent tensor from the encoder and processes it through a symmetric encoder-decoder structure. The backbone's encoder path progressively downsamples the spatial resolution while increasing the feature depth, capturing hierarchical features at different scales. The decoder path then upsamples the representation, using skip connections to merge high-resolution features from the corresponding encoder stages. This U-Net design allows the model to integrate both local and global spatial information effectively.

3\. **LoRA Integration**: Low-Rank Adaptation (LoRA) is integrated directly into the *WindowAttentio*n mechanism within the Swin Transformer backbone. This technique is highly relevant to our methodology of fine-tuning the converted pre-trained model. When enabled, small, trainable low-rank matrices (*lora_A* and *lora_B*) are added to the query-key-value (QKV) and projection layers of the attention blocks. During fine-tuning, the original model weights (which we have converted from PyTorch) are frozen, and only these low-rank matrices are updated. This significantly reduces the number of trainable parameters, making fine-tuning on specific tasks or datasets much more memory and computationally efficient. The LoRA weights are conditioned on the rollout step, allowing the model to learn distinct adaptations for different forecast horizons.

4\. **Decoder (*Perceiver3DDecoder*)**: The final stage reverses the encoding process. It receives the processed tensor from the backbone and uses another *PerceiverResampler* to function as a de-aggregator. It cross-attends from a set of query vectors (one for each atmospheric level) to the backbone's output, projecting the information back into the distinct atmospheric pressure levels. Finally, separate linear output heads are used for each surface and atmospheric variable to produce the final forecast.

### Code Implementation

This section details the practical implementation challenges and solutions we encountered while translating Aurora from PyTorch to JAX. These implementation details reveal the engineering complexity behind creating an efficient, large-scale foundation model training pipeline. Given the substantial codebase involves- spanning data loading, model architecture, training loops, and optimization strategies- this represented a considerable undertaking for a developer to implement and debug comprehensively.

**Explore the code**: <https://github.com/akshatkaush/aurora>

#### Hybrid DataLoader: Mixing PyTorch and JAX

One of the most pragmatic decisions in our implementation was to create a hybrid dataloader that leverages PyTorch's mature data loading infrastructure while outputting JAX arrays. Our *HresT0SequenceDataset* class inherits from PyTorch's *IterableDataset* but converts all data to JAX arrays:

class HresT0SequenceDataset(IterableDataset):\
    def __init__(self, zarr_path: str, mode: str = "train", shuffle: bool = True,\
                 seed: int | None = None, steps: int = 1):\
        # Use xarray for efficient zarr reading\
        ds_full = xr.open_zarr(zarr_path, consolidated=True, chunks={"time": 1})

        # Convert static variables to JAX arrays immediately\
        self.static_vars = {\
            "z": jnp.array(static_ds["z"].values[0]),\
            "slt": jnp.array(static_ds["slt"].values[0]),\
            "lsm": jnp.array(static_ds["lsm"].values[0]),\
        }

    def __iter__(self):\
        for i in idxs:\
            # Convert each variable to JAX arrays during iteration\
            surf_in = {\
                key: jnp.array(self.ds[var].isel(time=[i-2, i-1]).fillna(0).values[None])\
                for key, var in surf_map.items()\
            }

This hybrid approach was essential because:

1.  **PyTorch's DataLoader ecosystem**: We needed access to PyTorch's robust multiprocessing, batching, and shuffling capabilities.
2.  **JAX array format**: The model expects JAX arrays for all computations and transformations.
3.  **Memory efficiency**: Converting to JAX arrays early prevents unnecessary copies between frameworks.
4.  **Zarr integration**: PyTorch's ecosystem has better support for large-scale data formats like Zarr through xarray.
5.  **Optimization Opportunities**: While this hybrid approach provided a practical solution, it is not the most optimized implementation possible. There remains significant scope for parallelization in the data loading pipeline that should be further explored for efficiency improvements. Future work could investigate native JAX data loading solutions, and custom parallel preprocessing pipelines.

#### Strategic JIT Compilation

JAX's Just-In-Time (JIT) compilation is crucial for performance, but requires careful consideration of which functions to compile and how to handle dynamic vs. static arguments. Our implementation uses *@jax.jit* strategically throughout the codebase:

@partial(jax.jit, static_argnums=(4, 5))\
def train_step(state, inBatch: Batch, target_batches: List[Batch],\
               rng, steps: int, average_loss: bool):

The *static_argnums = (4, 5)* tells JAX that *steps* and *average_loss* are compile-time constants. This was critical because:

1.  **Loop unrolling**: JAX can unroll loops when the step count is static, leading to better optimization.
2.  **Conditional compilation**: The *average_loss* boolean enables different code paths to be optimized separately.
3.  **Memory planning**: Static step counts allow JAX to pre-allocate memory more efficiently.

**Model Component JIT in Architecture:**

self.encoder = nn.remat(\
    Perceiver3DEncoder,\
    static_argnums=(2, 3),  # patch_size, embed_dim are static\
)(...)

self.backbone = nn.remat(\
    Swin3DTransformerBackbone,\
    static_argnums=(1, 2, 3, 4, 5),  # All architectural params are static\
)(...)

Each component uses different *static_argnums* because they receive different static parameters during initialization. Getting these right was essential for successful compilation.

#### Gradient Checkpointing: Finding the Right Balance

**Strategic Placement:**

# Main model components - coarse-grained checkpointing\
self.encoder = nn.remat(Perceiver3DEncoder, static_argnums=(2, 3))(...)

self.encoder_layers = [\
      nn.remat(\
          Basic3DEncoderLayer,\
          static_argnums=(3, 4, 5, 6),  # res, rollout_step, training are static\
      )(\
          dim=int(self.embed_dim * 2**i_layer),\
          depth=self.encoder_depths[i_layer],\
          # ... other config parameters\
          downsample_temp=PatchMerging3D if i_layer < self.num_encoder_layers - 1 else None,\
      )\
      for i_layer in range(self.num_encoder_layers)\
  ]

        # Memory-optimized decoder layers with gradient checkpointing\
  self.decoder_layers = [\
      nn.remat(\
          Basic3DDecoderLayer,\
          static_argnums=(3, 4, 5, 6),  # res, pad_outs, rollout_step, training\
      )(\
          dim=int(self.embed_dim * 2**(self.num_decoder_layers - i_layer - 1)),\
          depth=self.decoder_depths[i_layer],\
          # ... other config parameters\
          upsample_temp=PatchSplitting3D if i_layer < self.num_decoder_layers - 1 else None,\
      )\
      for i_layer in range(self.num_decoder_layers)\
  ]\
self.decoder = nn.remat(Perceiver3DDecoder, static_argnums=(2, 3, 4, 5))(...)

**The Challenge of Optimal Placement:**

Finding the right granularity for gradient checkpointing was difficult because:

1\. **Too coarse**: Checkpointing only the main components didn't save enough memory for large rollouts.

2\. **Too fine**: Checkpointing every small operation caused excessive recomputation overhead.

3\. **Memory vs. Speed trade-off**: More checkpointing reduces memory but increases training time due to recomputation.

**Our Solution**: We implemented a hierarchical approach where components (*encoder*, *decoder*) are checkpointed at the top level, and *the backbone is checkpointed at the granular level at each different layer*. This provided the best balance of memory efficiency and computational overhead.

#### PyTree Compatibility: The Batch.py Conversion Challenge

Converting the *batch.py* module from PyTorch to JAX proved to be one of the most time-consuming implementation challenges. JAX and Flax operate under a strict PyTree paradigm, where all data structures must be composed of nested combinations of lists, tuples, dictionaries, and JAX arrays --- no custom classes with complex state management are allowed. The original PyTorch Batch class contained sophisticated serialization and deserialization logic, method-heavy interfaces, and stateful operations that fundamentally conflicted with JAX's functional programming requirements. I had to completely restructure the data organization, replacing class methods with pure functions, reimplementing all the normalization and cropping operations as standalone functions that return new Batch instances rather than modifying existing ones, and ensuring that every data transformation preserved the PyTree structure. The serialization logic was particularly complex because JAX requires explicit tree flattening and unflattening procedures, and getting the nested metadata handling to work correctly with operations like *jax.tree_util.tree_map* took extensive debugging and restructuring of how we handled the atmospheric and surface variable dictionaries.

#### **Forward Pass Validation: Point-wise Difference Analysis**

After completing the full forward pass implementation, verifying the correctness of our JAX translation required more than just calculating aggregate metrics like MAE and RMSE between PyTorch and JAX runs. To ensure our implementation was truly faithful to the original, we performed detailed point-wise difference analysis across all atmospheric and surface variables.

This validation approach involved running identical inputs through both the original PyTorch model and our JAX implementation, then computing pixel-by-pixel differences for each predicted variable. The resulting difference maps provided crucial insights into the spatial patterns of any discrepancies between the two implementations. For atmospheric variables (temperature, u-wind, v-wind, specific humidity, and geopotential height at various pressure levels) and surface variables (2m temperature, 10m winds, and mean sea level pressure), we generated comprehensive comparison visualizations showing Truth, PyTorch predictions, JAX predictions, and their point-wise differences.

![](https://cdn-images-1.medium.com/max/1440/1*C-KSZKibR4IHoqsmjPPIvw.png)

**Surface variables comparison (Truth, PyTorch, JAX, Differences)**

![](https://cdn-images-1.medium.com/max/1440/1*vpb90LkUx1WMwzjVzrUWSw.png)

**Atmospheric variables comparison (Truth, PyTorch, JAX, Differences)**

### Experimental Setup and Results

![](https://cdn-images-1.medium.com/max/1440/1*bbA57oYqmR0zF2r3qRePyg.png)

This section details the datasets, models, and training procedures used in our study. We present three distinct experimental setups to analyze different fine-tuning strategies for the Aurora model.

Our experiments use the HRES-T0 dataset, a subset of the high-resolution atmospheric model data from the European Centre for Medium-Range Weather Forecasts (ECMWF), made available through WeatherBench. For our purposes, data from the years 2020 and 2021 serve as the training set, while data from 2022 is reserved for validation. The codebase includes a script for downloading and preprocessing this data into the required format.

#### Setup 1: Baseline Single-Step Fine-Tuning

This initial experiment establishes a baseline by fine-tuning the model to predict the next single time step.

**Dataloade**r: We implemented a custom dataloader in JAX using pytorch dataloader's capability, HresT0SequenceDataset, which reads data from the Zarr store. It is designed to be efficiently yielding sequences of weather states for a given forecast lead time. For this baseline experiment, the dataloader was configured to provide an input state and a single target state, corresponding to a one-step-ahead prediction.

**Model and Training**: We used the `AuroraSmall` model configuration, which has an embedding dimension of 256 and encoder/decoder depths of (2, 6, 2). The model was initialized with the pre-trained weights converted from the official PyTorch release. Fine-tuning was performed on all model parameters using the AdamW optimizer with a learning rate of 5e-5 and a 1000 step warm-up schedule. The training process involved feeding the model a single input state and calculating the loss based on its single-step prediction against the ground truth.

**Training Result**:

The graph below shows how the model performed as we fine-tuned it for the single-step prediction task. You can clearly see the training loss steadily decreasing, which indicates that our setup --- with the chosen learning rate and optimizer --- worked effectively, helping the model learn quickly. The validation loss also follows the same downward trend, suggesting that the model isn't just memorizing but actually learning patterns that generalize well to new, unseen data. Given that predicting just one step ahead is relatively straightforward, it's reassuring to see the model rapidly adapting within the first few epochs. These results give us confidence that this approach provides a solid foundation for tackling more complex multi-step forecasting tasks in future experiments.

![](https://cdn-images-1.medium.com/max/1440/1*WUOFTc44xA-6cXwtRtucSQ.png)

Validation RMSE and MAE for one step fine tuning 

![](https://cdn-images-1.medium.com/max/1440/1*7OeYEICRWXR6gLbuAct44w.png)

Training RMSE, MAE, learning rate

#### Setup 2: Multi-Step Rollout Fine-Tuning

This setup explores the model's ability to learn from longer-term dependencies by training it to predict a sequence of two future states auto-regressively.

**Model and Training**: The same AuroraSmall model configuration and pre-trained weights were used as in the baseline. The key difference lies in the training loop. We performed a two-step rollout (rollout_steps=2). In this process, the model first predicts the state at time t+1 based on the input at time t. Then, this prediction is fed back into the model as input to generate a prediction for time *t+2*.

**Loss Calculation and Backpropagation**: To encourage the model to make accurate predictions at each step of the rollout, the loss was computed as the average of the losses from both prediction steps. That is, the loss for the prediction at *t+1* and the loss for the prediction at *t+2* were calculated independently against their respective ground truth states and then averaged. This combined loss was then used to compute the gradients, which were backpropagated through the entire two-step computation to update the model's weights.

**Training Result**:

The graph below highlights the model's performance during the two-step fine-tuning process. Compared to our baseline single-step training, we notice the training loss declines steadily but more gradually, reflecting the increased complexity of predicting two consecutive future states. The validation loss follows a similar trend, reinforcing the idea that the model successfully learns meaningful patterns that generalize to new data even in this more challenging task. It's encouraging to see the model's ability to maintain stable improvement across both steps of the rollout, suggesting it is effectively capturing longer-term dependencies between sequential predictions.

![](https://cdn-images-1.medium.com/max/1440/1*WCiPngxRuJS_8XtdrGSaGw.png)

Validation RMSE and MAE for two-step fine tuning

![](https://cdn-images-1.medium.com/max/1440/1*Fnf13PHe9W0OHXB2Zppa5w.png)

Training RMSE, MAE, learning rate for Two-Step fine tuning

#### Setup 3: LoRA Fine-Tuning with Multi-Step Rollout

This final and most detailed experiment investigates the effectiveness of Low-Rank Adaptation (LoRA) for parameter-efficient fine-tuning in the context of a three-step forecast.

**LoRA Integration and the "Push-Forward Trick**: In this setup, we enabled LoRA in the AuroraSmall model and froze the base model parameters by setting freeze_base=True. This means that only the small, low-rank adapter matrices in the attention layers of the Swin Transformer backbone were trained.

A critical component of this setup is the use of the "push-forward trick" for the multi-step rollout, implemented via jax.lax.stop_gradient. During the three-step rollout, after the model predicts the state for t+1, stop_gradient is applied to this prediction before it is fed back as input to predict the state for t+2. The same process is repeated for the prediction at t+2 before it is used to predict t+3. This prevents gradients from later steps from flowing back into the model through intermediate predictions. Consequently, the model cannot learn to "correct" a poor prediction at one step by adjusting its behavior at a subsequent step. Instead, it is forced to learn a robust and accurate single-step transition function, as the gradients for each step's prediction are calculated based only on the model's output at that specific step. This choice was made to make the model fit into GPU memory and to reduce its overall memory footprint.

**Loss Calculation and Gradient Flow**: The loss was computed as the average of the losses from all three prediction steps. However, due to the stop_gradient call and the frozen base model, the backpropagation process is fundamentally different. Gradients are calculated for the LoRA parameters at each step, but the frozen state of the base model ensures that its weights are not updated. This isolates the learning process entirely to the LoRA adapters, allowing for efficient fine-tuning of the model's behavior without altering the foundational knowledge stored in the pre-trained weights.

**Use Cases for LoRA**: The use of LoRA is particularly advantageous when computational resources are limited or when adapting a large, general-purpose foundation model to multiple specialized downstream tasks. By only training a small number of parameters, LoRA significantly reduces the memory footprint and training time required for fine-tuning, allowing the creation of multiple lightweight adapter models for different tasks that all share the same frozen base model. However, LoRA is not well suited to tasks that require a fundamental shift in the model's internal representations --- it makes low-rank adjustments to existing weights rather than learning entirely new features --- so for tasks with a large domain shift from the pre-training data, full fine-tuning may still be necessary to achieve state-of-the-art performance. According to the authors, applying LoRA yielded the best long-term mean squared error (MSE), albeit with slightly blurrier predictions, while omitting LoRA produced more realistic forecasts but incurred a marginally higher long-term MSE.

**Experimental Variants**: We conducted this LoRA experiment under two conditions to assess its robustness:

1\. **From Pre-trained Weights**: LoRA was applied directly to the *AuroraSmall* model initialized with the original converted pre-trained weights.

The graphs below illustrate the performance of the model when fine-tuned using LoRA with the three-step rollout approach. Overall, we notice that training stabilizes relatively quickly despite the additional complexity of longer forecasting horizons and parameter constraints. Because only the LoRA adapter parameters were updated --- while the base model remained frozen --- the training and validation losses decline more slowly compared to full fine-tuning, as the model must make efficient, targeted adaptations rather than broader changes.

![](https://cdn-images-1.medium.com/max/1440/1*uqcWPgh6MjUhkVGKNiNC8w.png)

LoRA fine tuning validation RMSE and MAE

![](https://cdn-images-1.medium.com/max/1440/1*GQKo4jGUa6L4kduFyYjjiw.png)

LoRA fine tuning training RMSE and MAE

Interestingly, employing the push-forward trick (stop gradients) forces the model to focus solely on learning accurate single-step transitions, without relying on adjustments at later steps. This resulted in a more stable and memory-efficient training process, ideal for GPU resource constraints. As suggested by the authors, LoRA-based fine-tuning effectively lowered long-term Mean Squared Error (MSE), although predictions became slightly blurrier compared to the full-parameter approach, which yielded sharper yet marginally less accurate forecasts.

![](https://cdn-images-1.medium.com/max/1440/1*-UpBoNaTFtmczwd4CVLmOw.png)

2\. **From 2-Step Fine-tuned Weights**: LoRA was applied to the model that had already been thoroughly fine-tuned for two rollout steps (the model resulting from Setup 2). This tests whether LoRA can still provide efficient adaptation on a model that has already undergone some specialization.

![](https://cdn-images-1.medium.com/max/1440/1*E9CMy9F6ADAMXAWJ93FfqQ.png)

#### **Visual Results Comparison**

To better understand the qualitative differences between our various fine-tuning approaches, we present visual comparisons of the model outputs across different training strategies. The following visualizations demonstrate the prediction quality and accuracy differences between our three experimental approaches. (These images have been pasted here in lower resolutions that is making the sharpness not so distinct.)

We compare results from four key experimental configurations:

-   **OneStepFineTuning**: Outputs from our baseline single-step fine-tuning approach (Setup 1)

![](https://cdn-images-1.medium.com/max/1440/1*ZRiggyL5V0YJblcCCWLCQw.png)

Surface variables after one step fine-tuning

![](https://cdn-images-1.medium.com/max/1440/1*uRJ1q2f__ORPKl44W6TB2w.png)

Atmospheric Variables after one step fine-tuning

-   **twostepFineTuning**: Results from our multi-step rollout fine-tuning (Setup 2)

![](https://cdn-images-1.medium.com/max/1440/1*mK7eHYi5wgRwfAoTpG7DmA.png)

Surface variables for two step fine-tuning

![](https://cdn-images-1.medium.com/max/1440/1*3kh5jI5EcBxRUaQTWK84hQ.png)

atmospheric variables for two step fine-tuning

-   **freezeBase LoRA stop gradients**: Results from our parameter-efficient LoRA fine-tuning with frozen base weights and stop gradient implementation (Setup 3)

![](https://cdn-images-1.medium.com/max/1440/1*BjKmDSN-not3-c-8AgRVCA.png)

Surface variables with LoRA fine-tuning with frozen base and stop gradients.

![](https://cdn-images-1.medium.com/max/1440/1*YH8O_KaQ3-9GW8hbocsmwQ.png)

Atmospheric variables with LoRA fine-tuning with frozen base and stop gradients.

-   **originalWeights**: Predictions using the original converted pre-trained weights without fine-tuning

![](https://cdn-images-1.medium.com/max/1440/1*nXv-p8uCcL3KblaGk0Ydvg.png)

Surface variables without fine-tuning

![](https://cdn-images-1.medium.com/max/1440/1*mv27ro7brES_bePGfKTVZw.png)

Atmospheric variables without fine-tuning

**Key Observations**

Our visual analysis reveals a clear progression in prediction quality and accuracy. Most notably, the results demonstrate improved performance in the following order: **one step > two step > original weights**. This ordering suggests that even minimal fine-tuning (single-step) provides substantial improvements over the base pre-trained model, while the two-step fine-tuning strikes an effective balance between adaptation and maintaining the model's foundational knowledge.

The single-step fine-tuned model produces the sharpest and most accurate short-term predictions, likely because it can focus entirely on optimizing immediate state transitions without the complexity of longer rollouts. The two-step fine-tuning shows robust performance across both prediction horizons, demonstrating the model's ability to learn from longer temporal dependencies while maintaining good accuracy. In contrast, the original weights, while still producing reasonable forecasts due to the strong pre-training, lack the task-specific adaptations that significantly enhance prediction quality.

The **freezeBase LoRA stop gradients** results provide particularly interesting insights into parameter-efficient fine-tuning. These visualizations demonstrate how LoRA adaptation with frozen base weights can achieve competitive prediction quality while using significantly fewer trainable parameters. The stop gradient implementation forces the model to learn robust single-step transitions, resulting in predictions that, while potentially slightly blurred compared to full fine-tuning, maintain strong temporal consistency and lower long-term MSE. This approach represents an optimal balance between computational efficiency and prediction accuracy, making it especially valuable for resource-constrained environments or when fine-tuning multiple task-specific adapters.

These visual results complement our quantitative metrics and provide intuitive evidence for the effectiveness of our fine-tuning strategies, particularly highlighting how targeted adaptation can substantially improve foundation model performance on specific forecasting tasks.

### Conclusion

This work represents the first comprehensive open-source implementation of Microsoft's Aurora weather forecasting model in JAX, providing the research community with complete training code, optimization strategies, and detailed performance benchmarks. Through our systematic translation from PyTorch to JAX and extensive experimental validation, we have demonstrated both the feasibility and the trade-offs inherent in adopting JAX for large-scale foundation model development in Earth system sciences.

Our implementation successfully reproduced the Aurora architecture's core functionality while revealing important insights about framework choice in deep learning research. While JAX's functional programming paradigm and XLA compilation offered theoretical advantages in terms of optimization potential and code clarity, our benchmarks showed that the PyTorch implementation maintained superior forward-pass performance in practice. This highlights the ongoing maturity gap between PyTorch's extensive ecosystem optimizations and JAX's more nascent but promising infrastructure. We think that there is still scope for optimizations, and the full use of JAX and its features is still left in this implementation. Nevertheless, JAX's composable transformation, particularly vmap for batching and grad for differentiation, significantly simplified our implementation and enhanced code maintainability.

The experimental results across our three fine-tuning setups provide valuable guidance for practitioners working with foundation models in atmospheric science. Our baseline single-step fine-tuning established that the converted model retained full functionality. At the same time, the multi-step rollout experiments revealed the model's capability to learn longer-term dependencies through autoregressive training. Most significantly, our LoRA-based fine-tuning with the push-forward trick demonstrated that parameter-efficient adaptation can achieve competitive performance while dramatically reducing computational requirements, a crucial finding for resource-constrained research environments.

The trade-offs we observed between different fine-tuning strategies reflect broader themes in foundation model adaptation. Full parameter fine-tuning produced sharper, more realistic forecasts but required substantial computational resources. In contrast, LoRA-based adaptation achieved lower long-term MSE with reduced memory footprint, albeit with slightly blurred predictions. The push-forward trick proved essential for memory efficiency in multi-step training, forcing the model to learn robust single-step transitions rather than relying on error correction across rollout steps.

Several limitations warrant acknowledgment. Our experiments were constrained to the AuroraSmall variant due to computational limitations, and the JAX implementation's performance lag suggests room for optimization through advanced JAX features such as custom kernels or more aggressive compiler optimizations. As this represents the foundational work of a single developer, future iterations will focus on efforts to fully leverage JAX's performance capabilities and close the remaining performance gaps with the original PyTorch implementation.

Looking forward, this work opens several promising research directions. The modular JAX implementation facilitates experimentation with alternative architectures, such as different attention mechanisms or novel temporal encoding schemes. The LoRA integration provides a foundation for multi-task learning across different Earth system domains, while the explicit state management in JAX enables sophisticated techniques like gradient checkpointing and model parallelism that could extend to larger model variants.

This is an ongoing research project with many open questions and opportunities for improvement. We actively invite collaboration from the broader research community, whether in optimizing the JAX implementation, extending Aurora to new Earth system domains, exploring novel fine-tuning techniques, or applying the model to specific forecasting challenges. We are open to engagement, from brief discussions and code reviews to long-term research. We encourage anyone interested in reproducing our results, extending the methodology, or exploring new use cases to reach out with questions and suggestions.

Perhaps most importantly, by making our complete implementation publicly available, we hope to lower the barriers for research groups seeking to build upon Aurora's foundation model approach. The atmospheric sciences community benefits significantly from standardized, well-documented implementations that enable rapid prototyping and collaborative development. Our work demonstrates that with careful engineering and appropriate optimization strategies, JAX can serve as a viable platform for large-scale Earth system modeling, contributing to the broader democratization of AI-driven climate research.

The successful translation and benchmarking of Aurora in JAX represents more than a technical exercise; it exemplifies the kind of open, reproducible research practices that will be essential as foundation models become increasingly central to scientific discovery in Earth system sciences.

### **Acknowledgments**

We sincerely thank the Microsoft Research team and the authors of the original Aurora paper for their groundbreaking work in developing this foundation model for Earth system forecasting. Their decision to publicly make the pre-trained model weights available has been instrumental in enabling this research and furthering the democratization of AI-driven Earth Science. We particularly appreciate their thorough documentation and the provision of the AuroraSmall variant, which made our experimental work computationally feasible.

We also acknowledge WeatherBench for providing access to the HRES-T0 dataset used in our experiments. Their efforts in curating and distributing high-quality atmospheric data have been essential for advancing machine learning research in meteorology and enabling reproducible benchmarking across the community.

We are grateful to Wessel Bruinsma for taking the time to assist with the verification of our implementation results. His careful review and feedback were instrumental in ensuring the correctness and reliability of our JAX translation.

Special thanks to Dr. Paris Perdikaris, whose guidance and expertise were invaluable throughout this project. His insights into the intersection of machine learning and scientific computing, as well as his continuous support and mentorship, were essential in navigating the technical challenges of this implementation and ensuring the scientific rigor of our approach.