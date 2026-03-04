#ifndef FOCUS_ENGINE_H
#define FOCUS_ENGINE_H

#ifdef __cplusplus
extern "C"
{
#endif
    enum ModelType
    {
        MODEL_HRNET = 0,
        MODEL_TRANSFORMER = 1,
        MODEL_ACTOR = 2
    };

    void *engine_init(const char *hrnet_path, const char *transformer_path, const char *actor_path);
    const float *engine_inference(void *handle, int model_type, const float *input_data, int input_size, int *out_size);
    void engine_free(void *handle);

#ifdef __cplusplus
}
#endif

#endif // FOCUS_ENGINE_H
