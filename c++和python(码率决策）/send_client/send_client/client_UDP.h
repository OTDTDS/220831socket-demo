#ifndef __client_UDP_h_
#define __client_UDP_h_

//#include <Python.h> 需要添加python.lib环境
#include<windows.h>
//#include <arrayobject.h>
//#include<cmath>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
//#pragma comment(lib,"User32.lib")
//#include "razor_api.h"
//#include "adapt_sender.h"


#define MIN_BIT_RATE	321
#define MAX_BIT_RATE	4280
#define FEEDBACK_DURATION	1000
#define MILLISECONDS_IN_SECOND	1000
#define S_INFO  6
#define S_LEN  8
#define A_DIM  6
#define TRAIN_SEQ_LEN 100
#define BIT_RATE_INTERVAL  50
#define M_IN_K 1000
#define REBUF_PENALTY  20  //4.0 / (6 / 30)
#define SMOOTH_PENALTY  1
#define DELAY_PENALTY   10 //4.0 / 0.

//QiIfffI
typedef struct
{
	uint64_t        iter_count;
	int32_t			sending_bit_rate;
	uint32_t		received_bit_rate;
	float			buffer_size;

	float			delay;
	float			packet_loss_rate;

	uint32_t		nack_sent_count;
}adapt_model_input_t;

//int32_t	            client_UDP(adapt_model_input_t i);

#endif