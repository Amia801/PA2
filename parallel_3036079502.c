/*
 * PLEASE WRITE DOWN FOLLOWING INFO BEFORE SUBMISSION
 * FILE NAME: parallel_3036079502.c
 * NAME: CHEN JUNJIE
 * UID: 3036079502
 * Development Platform: Ubuntu noble 24.04 x86_64 @AMD Ryzen 7 6800H @gcc version 13.2.0
 * Remark: All in 3
 * How to compile separately: (gcc -o parallel parallel_3036079502.c -O2 -lm -lpthread)
 */

#include "common.h" // some common definitions

#include <unistd.h>       // for nearly everything :)
#include <stdio.h>        // for printf, sprintf, fgets
#include <stdlib.h>       // for malloc, calloc
#include <stdint.h>       // for uint8_t and uint64_t
#include <time.h>         // for time
#include <string.h>       // for memcpy and strcmp
#include <sys/resource.h> // for rusage collection

#include "model.h" // for Llama definitions -> no need to know

int pos = 0;             // global position of generation
Transformer transformer; // transformer instance to be init
Tokenizer tokenizer;     // tokenizer instance to be init
Sampler sampler;         // sampler instance to be init

// YOUR CODE STARTS HERE
#include <pthread.h>
#include <semaphore.h> // uncomment this line if you use semaphore
#include <stdbool.h>   // uncomment this line if you want true / false

// you may define global variables here
int num_thr = 0;
int thr_no = 0;

// ----------------------------------------------------------------------------
// mat_vec_mul
typedef struct
{
    float *out;
    QuantizedTensor *vec;
    QuantizedTensor *mat;
    int col;
    int row;
    int start_row;
    int end_row;
} ThreadData_mat_vec_mul;
// mat_vec_mul
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// multi_head_attn
typedef struct
{
    float *out;
    float *q;
    float *key_cache;
    float *value_cache;
    float *att;
    int seq_len;
    int n_heads;
    int head_size;
    int kv_dim;
    int kv_mul;
    int start_head;
    int end_head;
} ThreadData_multi_head_attn;
// multi_head_attn
//  ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// thread_pool

#define MAX_TASKS 100

typedef struct
{
    void (*function)(void *);
    void *arg;
} task_t;

typedef struct
{
    pthread_t *threads;
    int num_thr;
    task_t task_queue[MAX_TASKS];
    int task_count;
    int task_front;
    int task_rear;
    int pending_tasks; 
    pthread_mutex_t lock;
    pthread_cond_t cond;
    pthread_cond_t tasks_done; 
    bool stop;
} thread_pool_t;

static thread_pool_t *pool = NULL;

// thread_pool
//  ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// count_time
typedef struct {
    struct timeval user_time;
    struct timeval sys_time;
} ThreadTime;
// count_time
//  ----------------------------------------------------------------------------

// function executed by each thread to complete mat_vec_mul
// @note: please modify the signature to what you want
//same to the seq one
void mat_vec_mul_task_func(void *arg)
{   
    ThreadData_mat_vec_mul *data = (ThreadData_mat_vec_mul *)arg;
    float *out = data->out;
    QuantizedTensor *vec = data->vec;
    QuantizedTensor *mat = data->mat;
    int col = data->col;
    int start_row = data->start_row;
    int end_row = data->end_row;

    for (int i = start_row; i < end_row; i++)
    {
        float val = 0.0f; 
        int32_t ival = 0; 
        int in = i * col;

        for (int j = 0; j <= col - GS; j += GS)
        {
            for (int k = 0; k < GS; k++)
            {
                ival += ((int32_t)vec->q[j + k]) * ((int32_t)mat->q[in + j + k]);
            }
            val += ((float)ival) * mat->s[(in + j) / GS] * vec->s[j / GS];
            ival = 0;
        }

        out[i] = val;
    }
    return NULL;
}

// function executed by each thread to complete multi_head_attn
// @note: please modify the signature to what you want
// same to the seq one
void multi_head_attn_task_func(void *arg)
{
    ThreadData_multi_head_attn *data = (ThreadData_multi_head_attn *)arg;
    float *out = data->out;
    float *q = data->q;
    float *key_cache = data->key_cache;
    float *value_cache = data->value_cache;
    float *att = data->att;
    int seq_len = data->seq_len;
    int head_size = data->head_size;
    int kv_dim = data->kv_dim;
    int kv_mul = data->kv_mul;
    int start_head = data->start_head;
    int end_head = data->end_head;

    for (int h = start_head; h < end_head; h++)
    {
        float *head_q = q + h * head_size;
        float *head_att = att + h * seq_len;

        for (int t = 0; t <= pos; t++)
        {
            float *head_k = key_cache + t * kv_dim + (h / kv_mul) * head_size;
            float score = 0.0f;
            for (int i = 0; i < head_size; i++)
            {
                score += head_q[i] * head_k[i];
            }
            score /= sqrtf(head_size);
            head_att[t] = score;
        }

        softmax(head_att, pos + 1);

        float *head_out = out + h * head_size;
        memset(head_out, 0, head_size * sizeof(float));
        for (int t = 0; t <= pos; t++)
        {
            float *head_v = value_cache + t * kv_dim + (h / kv_mul) * head_size;
            float a = head_att[t];
            for (int i = 0; i < head_size; i++)
            {
                head_out[i] += a * head_v[i];
            }
        }
    }
    return NULL;
}

// ---------------------------------------------------------------------------- timing
void get_thread_times(ThreadTime *time) {
    struct rusage usage;
    getrusage(RUSAGE_THREAD, &usage); 
    time->user_time = usage.ru_utime;
    time->sys_time = usage.ru_stime;
}

double time_diff(struct timeval start, struct timeval end) {
    return (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1000000.0;
}
// ---------------------------------------------------------------------------- timing

// thread function used in pthread_create
// @note: YOU CAN NOT MODIFY this FUNCTION SIGNATURE!!!
void *thr_func(void *arg)
{   
    // thread number
    int selfnum = thr_no;
    thr_no++;
    // get thread start time
    ThreadTime start_time, end_time;
    get_thread_times(&start_time);


    while (true)
    {
        pthread_mutex_lock(&pool->lock);
        
        // wait for task
        while (pool->task_count == 0 && !pool->stop)
        {
            pthread_cond_wait(&pool->cond, &pool->lock);
        }
        // check if thread should stop
        if (pool->stop && pool->task_count == 0)
        {
            pthread_mutex_unlock(&pool->lock);
            break;
        }

        task_t task = pool->task_queue[pool->task_front];
        pool->task_front = (pool->task_front + 1) % MAX_TASKS;
        pool->task_count--;

        pthread_mutex_unlock(&pool->lock);

        // execute task
        task.function(task.arg);

        pthread_mutex_lock(&pool->lock);
        pool->pending_tasks--;
        // check if all tasks are done
        if (pool->pending_tasks == 0)
        {
            pthread_cond_signal(&pool->tasks_done);
        }
        pthread_mutex_unlock(&pool->lock);
    }

    // get thread end time
    get_thread_times(&end_time);
    double user_time = time_diff(start_time.user_time, end_time.user_time);
    double sys_time = time_diff(start_time.sys_time, end_time.sys_time);
    printf("Thread ID: %d, User Time: %.6f s, System Time: %.6f s\n", selfnum, user_time, sys_time);
    return NULL;
}

// function to initialize thread pool
// @note: YOU CAN NOT MODIFY this FUNCTION SIGNATURE!!!
void init_thr_pool(int num_thr)
{
    pool = (thread_pool_t *)malloc(sizeof(thread_pool_t));
    pool->num_thr = num_thr;
    pool->task_count = 0;
    pool->task_front = 0;
    pool->task_rear = 0;
    pool->pending_tasks = 0;
    pool->stop = false;
    pool->threads = (pthread_t *)malloc(num_thr * sizeof(pthread_t));

    pthread_mutex_init(&pool->lock, NULL);
    pthread_cond_init(&pool->cond, NULL);
    pthread_cond_init(&pool->tasks_done, NULL);

    // create threads
    for (int i = 0; i < num_thr; i++)
    {
        pthread_create(&pool->threads[i], NULL, thr_func, NULL);
    }
}

// function to close thread pool
// @note: YOU CAN NOT MODIFY this FUNCTION SIGNATURE!!!
void close_thr_pool()
{
    pthread_mutex_lock(&pool->lock);
    // set stop flag
    pool->stop = true;

    pthread_cond_broadcast(&pool->cond);
    pthread_mutex_unlock(&pool->lock);

    // wait for all threads to finish
    for (int i = 0; i < pool->num_thr; i++)
    {
        pthread_join(pool->threads[i], NULL);
    }

    // free resources
    pthread_mutex_destroy(&pool->lock);
    pthread_cond_destroy(&pool->cond);
    pthread_cond_destroy(&pool->tasks_done);
    free(pool->threads);
    free(pool);
    pool = NULL;
}

void add_task_to_pool(void (*function)(void *), void *arg)
{
    pthread_mutex_lock(&pool->lock);

    if (pool->task_count < MAX_TASKS)
    {   
        // add task to queue
        pool->task_queue[pool->task_rear].function = function;
        pool->task_queue[pool->task_rear].arg = arg;
        pool->task_rear = (pool->task_rear + 1) % MAX_TASKS;
        pool->task_count++;
        pool->pending_tasks++;

        pthread_cond_signal(&pool->cond); 
    }

    pthread_mutex_unlock(&pool->lock);
}

void wait_for_all_tasks()// wait for all tasks to complete
{   
    pthread_mutex_lock(&pool->lock);
    while (pool->pending_tasks > 0)
    {
        pthread_cond_wait(&pool->tasks_done, &pool->lock);
    }
    pthread_mutex_unlock(&pool->lock);
}
// ----------------------------------------------------------------------------
// entry function for multi-threading matrix multiplication
// @note: YOU CAN NOT MODIFY this FUNCTION SIGNATURE!!!

void mat_vec_mul(float *out, QuantizedTensor *vec, QuantizedTensor *mat, int col, int row)
{
    pthread_t threads[num_thr];
    ThreadData_mat_vec_mul thread_data[num_thr];
    int rows_per_thread = row / num_thr;

    for (int t = 0; t < num_thr; t++)
    {
        // initialize thread data
        thread_data[t].out = out;
        thread_data[t].vec = vec;
        thread_data[t].mat = mat;
        thread_data[t].col = col;
        thread_data[t].row = row;
        thread_data[t].start_row = t * rows_per_thread;
        thread_data[t].end_row = (t == num_thr - 1) ? row : (t + 1) * rows_per_thread;

        // add task to thread pool
        add_task_to_pool(mat_vec_mul_task_func, (void *)&thread_data[t]);
    }

    // wait for all tasks to complete
    wait_for_all_tasks();
}

// ----------------------------------------------------------------------------
// entry function for multi-threading multi-head-attention
// @note: YOU CAN NOT MODIFY FUNCTION SIGNATURE!!!
void multi_head_attn(
    float *out,         // output tensor [head, head_size]
    float *q,           // query tensor  [head, head_size]
    float *key_cache,   // cache of history key tensor   [kv_head, seq_len, head_size]
    float *value_cache, // cache of history value tensor [kv_head, seq_len, head_size]
    float *att,         // buffer for attention score [head, seq_len]
    int seq_len,        // current sequence length
    int n_heads,        // number of heades
    int head_size,      // size of each head
    int kv_dim,
    int kv_mul)
{
    // multihead attention. iterate over all heads
    pthread_t threads[num_thr];
    ThreadData_multi_head_attn thread_data[num_thr];
    int heads_per_thread = n_heads / num_thr;

    for (int t = 0; t < num_thr; t++)
    {
        // initialize thread data
        thread_data[t].out = out;
        thread_data[t].q = q;
        thread_data[t].key_cache = key_cache;
        thread_data[t].value_cache = value_cache;
        thread_data[t].att = att;
        thread_data[t].seq_len = seq_len;
        thread_data[t].n_heads = n_heads;
        thread_data[t].head_size = head_size;
        thread_data[t].kv_dim = kv_dim;
        thread_data[t].kv_mul = kv_mul;
        thread_data[t].start_head = t * heads_per_thread;
        thread_data[t].end_head = (t == num_thr - 1) ? n_heads : (t + 1) * heads_per_thread;
        // add task to thread pool
        add_task_to_pool(multi_head_attn_task_func, (void *)&thread_data[t]);
    }
    // wait for all tasks to complete
    wait_for_all_tasks();
}
// YOUR CODE ENDS HERE

// ----------------------------------------------------------------------------
// forward Transformer, you're not allowed to modify this part
float *forward(Transformer *transformer, int token, int pos)
{

    // a few convenience variables
    Config *p = &transformer->config;
    TransformerWeights *w = &transformer->weights;
    RunState *s = &transformer->state;
    float *x = s->x;
    int dim = p->dim;
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    int kv_mul = p->n_heads / p->n_kv_heads; // integer multiplier of the kv sharing in multiquery
    int hidden_dim = p->hidden_dim;
    int head_size = dim / p->n_heads;

    // copy the token embedding into x
    memcpy(x, w->token_embedding_table + token * dim, dim * sizeof(float));

    // forward all the layers
    for (int l = 0; l < p->n_layers; l++)
    {

        // attention rmsnorm
        rmsnorm(s->xb, x, w->rms_att_weight + l * dim, dim);

        // qkv matmuls for this position
        quantize(&s->xq, s->xb, dim);
        mat_vec_mul(s->q, &s->xq, w->wq + l, dim, dim);
        mat_vec_mul(s->k, &s->xq, w->wk + l, dim, kv_dim);
        mat_vec_mul(s->v, &s->xq, w->wv + l, dim, kv_dim);

        // RoPE relative positional encoding: complex-valued rotate q and k in each head
        for (int i = 0; i < dim; i += 2)
        {
            int head_dim = i % head_size;
            float freq = 1.0f / powf(10000.0f, head_dim / (float)head_size);
            float val = pos * freq;
            float fcr = cosf(val);
            float fci = sinf(val);
            int rotn = i < kv_dim ? 2 : 1; // how many vectors? 2 = q & k, 1 = q only
            for (int v = 0; v < rotn; v++)
            {
                float *vec = v == 0 ? s->q : s->k; // the vector to rotate (query or key)
                float v0 = vec[i];
                float v1 = vec[i + 1];
                vec[i] = v0 * fcr - v1 * fci;
                vec[i + 1] = v0 * fci + v1 * fcr;
            }
        }

        // save key,value at this time step (pos) to our kv cache
        int loff = l * p->seq_len * kv_dim; // kv cache layer offset for convenience
        float *key_cache_row = s->key_cache + loff + pos * kv_dim;
        float *value_cache_row = s->value_cache + loff + pos * kv_dim;
        memcpy(key_cache_row, s->k, kv_dim * sizeof(*key_cache_row));
        memcpy(value_cache_row, s->v, kv_dim * sizeof(*value_cache_row));

        multi_head_attn(s->xb, s->q, s->key_cache + loff, s->value_cache + loff, s->att,
                        p->seq_len, p->n_heads, head_size, kv_dim, kv_mul);

        // final matmul to get the output of the attention
        quantize(&s->xq, s->xb, dim);
        mat_vec_mul(s->xb2, &s->xq, w->wo + l, dim, dim);

        // residual connection back into x
        for (int i = 0; i < dim; i++)
        {
            x[i] += s->xb2[i];
        }

        // ffn rmsnorm
        rmsnorm(s->xb, x, w->rms_ffn_weight + l * dim, dim);

        // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
        // first calculate self.w1(x) and self.w3(x)
        quantize(&s->xq, s->xb, dim);
        mat_vec_mul(s->hb, &s->xq, w->w1 + l, dim, hidden_dim);
        mat_vec_mul(s->hb2, &s->xq, w->w3 + l, dim, hidden_dim);

        // SwiGLU non-linearity
        for (int i = 0; i < hidden_dim; i++)
        {
            float val = s->hb[i];
            // silu(x)=x*σ(x), where σ(x) is the logistic sigmoid
            val *= (1.0f / (1.0f + expf(-val)));
            // elementwise multiply with w3(x)
            val *= s->hb2[i];
            s->hb[i] = val;
        }

        // final matmul to get the output of the ffn
        quantize(&s->hq, s->hb, hidden_dim);
        mat_vec_mul(s->xb, &s->hq, w->w2 + l, hidden_dim, dim);

        // residual connection
        for (int i = 0; i < dim; i++)
        {
            x[i] += s->xb[i];
        }
    }

    // final rmsnorm
    rmsnorm(x, x, w->rms_final_weight, dim);

    // classifier into logits
    quantize(&s->xq, x, dim);
    mat_vec_mul(s->logits, &s->xq, w->wcls, dim, p->vocab_size);
    return s->logits;
}

// ----------------------------------------------------------------------------
// generation loop, you're not allowed to modify this part
void generate(char *prompt)
{
    // encode the (string) prompt into tokens sequence
    int num_prompt_tokens = 0;
    int *prompt_tokens = (int *)malloc((strlen(prompt) + 6) * sizeof(int)); // +6 reserved for prompt template
    encode(&tokenizer, prompt, prompt_tokens, &num_prompt_tokens);
    if (num_prompt_tokens < 1)
    {
        fprintf(stderr, "something is wrong, expected at least 1 prompt token\n");
        exit(EXIT_FAILURE);
    }

    // start the main loop
    int next;                     // place holder for next token
    int token = prompt_tokens[0]; // place holder of prev token, kickoff as prompt_tokens[0]
    int end_pos = pos + MAX_NEW_TOKENS + num_prompt_tokens;
    int start_pos = pos;
    long start_time = 0; // to be lazy iniialzied
    while (pos < end_pos)
    {

        // forward the transformer to get logits for the next token
        float *logits = forward(&transformer, token, pos);

        if (pos < start_pos + num_prompt_tokens - 1)
        {
            // if we are still processing the input prompt, force the next prompt token
            next = prompt_tokens[pos - start_pos + 1];
        }
        else if (pos == end_pos - 2)
        {
            // reaching the end, force it to close by <|im_end|>
            next = 2; // := <|im_end|>
        }
        else
        {
            // otherwise sample the next token from the logits
            next = sample(&sampler, logits);
        }

        pos++;

        // print the token as string, decode it with the Tokenizer object
        char *piece = decode(&tokenizer, token, next);
        if (pos >= num_prompt_tokens)
        {
            safe_printf(piece); // same as printf("%s", piece), but skips "unsafe" bytes
            fflush(stdout);
        }

        token = next;

        // init the timer here because the first iteration can be slower
        if (start_time == 0)
        {
            start_time = time_in_ms();
        }
    }
    printf("\n");

    long end_time = time_in_ms();
    // \033[0;32m set color to green and \033[0m reset to default, they won't be generate by LLM
    fprintf(stdout, "\033[0;32mlength: %d, speed (tok/s): %.4f \033[0m\n",
            pos, (pos - start_pos) / (float)(end_time - start_time) * 1000);

    free(prompt_tokens);
}

int main(int argc, char *argv[])
{
    ThreadTime start_time123, end_time123;
    get_thread_times(&start_time123);
    // default parameters
    char *model_path = "model.bin"; // e.g. out/model.bin
    char *tokenizer_path = "tokenizer.bin";
    float temperature = 0.6f; // 0.0 = greedy deterministic. 1.0 = original. don't set higher
    float topp = 0.9f;        // top-p in nucleus sampling. 1.0 = off. 0.9 works well, but slower
    char *prompt = NULL;      // prompt strings
    int num_prompt = 0;       // number of prompts
    uint64_t rng_seed = 0;    // seed rng with time by default
    // int num_thr          = 0;

    if (argc == 4)
    {
        num_thr = atoi(argv[1]);
        rng_seed = atoi(argv[2]);
        prompt = argv[3];
    }
    else
    {
        fprintf(stderr, "Usage:   ./seq <num_thr> <seed> <prompt>\n");
        fprintf(stderr, "Example: ./seq 4 42 \"What is Fibonacci Number?\"\n");
        fprintf(stderr, "Note:    <prompt> must be quoted with \"\", only one prompt supported\n");
        exit(1);
    }

    // parameter validation/overrides
    if (num_thr <= 0 || num_thr > 16)
    {
        fprintf(stderr, "num_thr must between 1 and 16 \n");
        exit(EXIT_FAILURE);
    }
    if (rng_seed <= 0)
        rng_seed = (unsigned int)time(NULL);

    // build the Transformer via the model .bin file
    build_transformer(&transformer, model_path);
    // build the Tokenizer via the tokenizer .bin file
    build_tokenizer(&tokenizer, tokenizer_path, transformer.config.vocab_size);
    // build the Sampler
    build_sampler(&sampler, transformer.config.vocab_size, temperature, topp, rng_seed);

    // initialize thread pool
    init_thr_pool(num_thr);

    printf("user: %s \n", prompt);
    // perform multi-threading generation
    generate(prompt);

    // close thread pool
    close_thr_pool();

    // memory and file handles cleanup
    free_sampler(&sampler);
    free_tokenizer(&tokenizer);
    free_transformer(&transformer);

    get_thread_times(&end_time123);
    double user_time = time_diff(start_time123.user_time, end_time123.user_time);
    double sys_time = time_diff(start_time123.sys_time, end_time123.sys_time);
    printf("Thread ID: main, User Time: %.6f s, System Time: %.6f s\n", user_time, sys_time);
    return 0;
}