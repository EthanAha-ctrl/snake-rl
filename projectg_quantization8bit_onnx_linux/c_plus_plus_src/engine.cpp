#include "engine.h"
#include <iostream>
#include <onnxruntime_cxx_api.h>
#include <vector>


struct EngineContext
{
    Ort::Env env;
    Ort::SessionOptions session_options;

    Ort::Session *hrnet_session;
    Ort::Session *transformer_session;
    Ort::Session *actor_session;

    Ort::MemoryInfo memory_info;

    // Output cache pre-allocated arrays to guarantee pointer stability in C API
    std::vector<float> hrnet_output_cache;
    std::vector<float> transformer_output_cache;
    std::vector<float> actor_output_cache;

    EngineContext(const char *hrnet_path, const char *transformer_path, const char *actor_path) : env(ORT_LOGGING_LEVEL_WARNING, "Focus_Engine"), hrnet_session(nullptr), transformer_session(nullptr), actor_session(nullptr), memory_info(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault))
    {
        // 8 threads matching the previous inference configuration on Apple Silicon/AArch64
        session_options.SetIntraOpNumThreads(8);
        session_options.SetInterOpNumThreads(8);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

        hrnet_session = new Ort::Session(env, hrnet_path, session_options);
        transformer_session = new Ort::Session(env, transformer_path, session_options);
        actor_session = new Ort::Session(env, actor_path, session_options);
    }

    ~EngineContext()
    {
        if (hrnet_session)
            delete hrnet_session;
        if (transformer_session)
            delete transformer_session;
        if (actor_session)
            delete actor_session;
    }
};

extern "C"
{

    EngineHandle engine_init(const char *hrnet_path, const char *transformer_path, const char *actor_path)
    {
        try
        {
            EngineContext *ctx = new EngineContext(hrnet_path, transformer_path, actor_path);
            std::cout << "[Focus Engine] Initialized successfully." << std::endl;
            return static_cast<EngineHandle>(ctx);
        }
        catch (const std::exception &e)
        {
            std::cerr << "[Focus Engine] Init failed: " << e.what() << std::endl;
            return nullptr;
        }
    }

    const float *engine_inference(EngineHandle handle, int model_type, const float *input_data, int input_size, int *out_size)
    {
        if (!handle)
        {
            *out_size = 0;
            return nullptr;
        }
        EngineContext *ctx = static_cast<EngineContext *>(handle);

        try
        {
            if (model_type == MODEL_HRNET)
            {
                std::vector<int64_t> input_shape = {1, 1, 480, 640};
                Ort::Value input_tensor = Ort::Value::CreateTensor<float>(ctx->memory_info, const_cast<float *>(input_data), input_size, input_shape.data(), input_shape.size());

                const char *input_names[] = {"image_input"};
                const char *output_names[] = {"vision_features"};

                auto output_tensors = ctx->hrnet_session->Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1);

                float *out_arr = output_tensors[0].GetTensorMutableData<float>();
                size_t count = output_tensors[0].GetTensorTypeAndShapeInfo().GetElementCount();

                ctx->hrnet_output_cache.assign(out_arr, out_arr + count);
                *out_size = static_cast<int>(count);
                return ctx->hrnet_output_cache.data();
            }
            else if (model_type == MODEL_TRANSFORMER)
            {
                std::vector<int64_t> input_shape = {1, static_cast<int64_t>(input_size)};
                Ort::Value input_tensor = Ort::Value::CreateTensor<float>(ctx->memory_info, const_cast<float *>(input_data), input_size, input_shape.data(), input_shape.size());

                const char *input_names[] = {"obs_stack"};
                const char *output_names[] = {"encoded_features"};

                auto output_tensors = ctx->transformer_session->Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1);

                float *out_arr = output_tensors[0].GetTensorMutableData<float>();
                size_t count = output_tensors[0].GetTensorTypeAndShapeInfo().GetElementCount();

                ctx->transformer_output_cache.assign(out_arr, out_arr + count);
                *out_size = static_cast<int>(count);
                return ctx->transformer_output_cache.data();
            }
            else if (model_type == MODEL_ACTOR)
            {
                std::vector<int64_t> input_shape = {1, static_cast<int64_t>(input_size)};
                Ort::Value input_tensor = Ort::Value::CreateTensor<float>(ctx->memory_info, const_cast<float *>(input_data), input_size, input_shape.data(), input_shape.size());

                const char *input_names[] = {"features"};
                const char *output_names[] = {"action"};

                auto output_tensors = ctx->actor_session->Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1);

                float *out_arr = output_tensors[0].GetTensorMutableData<float>();
                size_t count = output_tensors[0].GetTensorTypeAndShapeInfo().GetElementCount();

                ctx->actor_output_cache.assign(out_arr, out_arr + count);
                *out_size = static_cast<int>(count);
                return ctx->actor_output_cache.data();
            }
        }
        catch (const std::exception &e)
        {
            std::cerr << "[Focus Engine] Inference failed for model " << model_type << ": " << e.what() << std::endl;
            *out_size = 0;
            return nullptr;
        }

        *out_size = 0;
        return nullptr;
    }

    void engine_free(EngineHandle handle)
    {
        if (handle)
        {
            delete static_cast<EngineContext *>(handle);
            std::cout << "[Focus Engine] Freed resources." << std::endl;
        }
    }

} // extern "C"
