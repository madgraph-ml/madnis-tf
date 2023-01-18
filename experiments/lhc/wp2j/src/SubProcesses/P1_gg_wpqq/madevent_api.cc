#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"

#include <array>
#include <vector>
#include <iostream>

using namespace tensorflow;

extern "C" {
    void configure_code_tf_(bool, bool, double&); 
    void madevent_api_(double*, int&, int&, bool&, double&);
    void get_momenta_(double*);
    void get_multichannel_(double*);
}

class MGInterface {
    public:
        void Configure() { 
            double dconfig = 1.0;
            static bool configured{false};
            if(!configured) configure_code_tf_(true, true, dconfig);
            configured = true;
            return;
        }
        double CallMadgraph(double *rans, int ndim, int channel) { 
            double result;
            bool cut = true;
            madevent_api_(rans, ndim, channel, cut, result);
            return result;
        }
        std::vector<double> Momenta(int npart) {
            std::vector<double> momenta(npart*4);
            get_momenta_(momenta.data());
            return momenta;
        }
        std::vector<double> Alpha(int nchannels) {
            std::vector<double> alphas(nchannels);
            get_multichannel_(alphas.data());
            return alphas;
        }
        static MGInterface Instance() {
            static MGInterface interface;
            return interface;
        }
};

REGISTER_OP("CallMadgraph")
    .Attr("npart: int")
    .Attr("nchannels: int")
    .Input("rans: float64")
    .Input("channel: int32")
    .Output("wgt: float64")
    .Output("mom: float64")
    .Output("alpha: float64");

class CallMadgraphOp : public OpKernel {
    public:
        explicit CallMadgraphOp(OpKernelConstruction *context) : OpKernel(context) {
            // Get the number of particles
            OP_REQUIRES_OK(context,
                           context->GetAttr("npart", &npart_));
            // Get the number of channels
            OP_REQUIRES_OK(context,
                           context->GetAttr("nchannels", &nchannels_));
            // Check that npart is greater than 3
            OP_REQUIRES(context, npart_ > 3,
                        errors::InvalidArgument("Need at least 4 particles, got ",
                                                   npart_));
            MGInterface::Instance().Configure();
        }

        void Compute(OpKernelContext *context) override {
            // Grab the input tensors and their shapes
            const Tensor& input_tensor = context->input(0);
            auto input = input_tensor.flat<double>();
            TensorShape input_shape = input_tensor.shape();
            size_t nBatch = input_shape.dim_size(0);
            size_t nRandom = input_shape.dim_size(1);

            const Tensor& input_channel = context->input(1);
            auto in_channel = input_channel.flat<int>();
            TensorShape input_channel_shape = input_channel.shape();
            if(nBatch != input_channel_shape.dim_size(0))
                throw std::runtime_error("Random and Channel batch have different size");

            // Create a weight tensor 
            Tensor *weight_tensor = nullptr;
            TensorShape weight_shape = input_shape;
            weight_shape.RemoveDim(1);
            OP_REQUIRES_OK(context, context->allocate_output(0, weight_shape, &weight_tensor));
            auto weight_flat = weight_tensor->flat<double>();

            // Create a momenta tensor 
            Tensor *momenta_tensor = nullptr;
            TensorShape momenta_shape = input_shape;
            momenta_shape.RemoveDim(1);
            // Shape should be nbatch, npart, (E, px, py, pz)
            momenta_shape.AddDim(npart_);
            momenta_shape.AddDim(4);
            OP_REQUIRES_OK(context, context->allocate_output(1, momenta_shape, &momenta_tensor));
            auto momenta_flat = momenta_tensor->flat<double>();

            // Create a alpha tensor 
            Tensor *alpha_tensor = nullptr;
            TensorShape alpha_shape = input_shape;
            alpha_shape.RemoveDim(1);
            // Shape should be nbatch, nchannel
            alpha_shape.AddDim(nchannels_);
            OP_REQUIRES_OK(context, context->allocate_output(2, alpha_shape, &alpha_tensor));
            auto alpha_flat = alpha_tensor->flat<double>();

            // Calculate the cross sections
            double *rans = new double[nRandom];
            for(size_t ibatch = 0; ibatch < nBatch; ++ibatch) {
                for(size_t iran = 0; iran < nRandom; ++iran) {
                    rans[iran] = input(ibatch*nRandom + iran);
                }
                // Call API and obtain bare weight
                int channel = in_channel(ibatch);
                double weight = MGInterface::Instance().CallMadgraph(rans, nRandom, channel + 1);

                // Get momenta
                auto mom = MGInterface::Instance().Momenta(npart_);
                for(size_t imom = 0; imom < 4*npart_; ++imom){
                    momenta_flat(ibatch*4*npart_ + imom) = mom[imom];
                }

                // Get corresponding alpha and make sure its not NaN
                auto alpha = MGInterface::Instance().Alpha(nchannels_);
                for(size_t ialpha = 0; ialpha < nchannels_; ++ialpha){
                    if (std::isnan(alpha[ialpha]) || weight == 0.0) {
                        alpha_flat(ibatch*nchannels_ + ialpha) = 1.0 / nchannels_;
                    } else {
                        alpha_flat(ibatch*nchannels_ + ialpha) = alpha[ialpha];
                    }
                }

                // Return correct weigt
                weight_flat(ibatch) = weight/alpha_flat(ibatch*nchannels_ + channel);
            }
            delete[] rans;
        }

        ~CallMadgraphOp() {}
    private:
        int npart_;
        int nchannels_;
};

REGISTER_KERNEL_BUILDER(Name("CallMadgraph").Device(DEVICE_CPU), CallMadgraphOp);
