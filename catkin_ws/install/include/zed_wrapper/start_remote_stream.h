// Generated by gencpp from file zed_wrapper/start_remote_stream.msg
// DO NOT EDIT!


#ifndef ZED_WRAPPER_MESSAGE_START_REMOTE_STREAM_H
#define ZED_WRAPPER_MESSAGE_START_REMOTE_STREAM_H

#include <ros/service_traits.h>


#include <zed_wrapper/start_remote_streamRequest.h>
#include <zed_wrapper/start_remote_streamResponse.h>


namespace zed_wrapper
{

struct start_remote_stream
{

typedef start_remote_streamRequest Request;
typedef start_remote_streamResponse Response;
Request request;
Response response;

typedef Request RequestType;
typedef Response ResponseType;

}; // struct start_remote_stream
} // namespace zed_wrapper


namespace ros
{
namespace service_traits
{


template<>
struct MD5Sum< ::zed_wrapper::start_remote_stream > {
  static const char* value()
  {
    return "c9f6f4c6411b7a0c79a7a7357650993c";
  }

  static const char* value(const ::zed_wrapper::start_remote_stream&) { return value(); }
};

template<>
struct DataType< ::zed_wrapper::start_remote_stream > {
  static const char* value()
  {
    return "zed_wrapper/start_remote_stream";
  }

  static const char* value(const ::zed_wrapper::start_remote_stream&) { return value(); }
};


// service_traits::MD5Sum< ::zed_wrapper::start_remote_streamRequest> should match 
// service_traits::MD5Sum< ::zed_wrapper::start_remote_stream > 
template<>
struct MD5Sum< ::zed_wrapper::start_remote_streamRequest>
{
  static const char* value()
  {
    return MD5Sum< ::zed_wrapper::start_remote_stream >::value();
  }
  static const char* value(const ::zed_wrapper::start_remote_streamRequest&)
  {
    return value();
  }
};

// service_traits::DataType< ::zed_wrapper::start_remote_streamRequest> should match 
// service_traits::DataType< ::zed_wrapper::start_remote_stream > 
template<>
struct DataType< ::zed_wrapper::start_remote_streamRequest>
{
  static const char* value()
  {
    return DataType< ::zed_wrapper::start_remote_stream >::value();
  }
  static const char* value(const ::zed_wrapper::start_remote_streamRequest&)
  {
    return value();
  }
};

// service_traits::MD5Sum< ::zed_wrapper::start_remote_streamResponse> should match 
// service_traits::MD5Sum< ::zed_wrapper::start_remote_stream > 
template<>
struct MD5Sum< ::zed_wrapper::start_remote_streamResponse>
{
  static const char* value()
  {
    return MD5Sum< ::zed_wrapper::start_remote_stream >::value();
  }
  static const char* value(const ::zed_wrapper::start_remote_streamResponse&)
  {
    return value();
  }
};

// service_traits::DataType< ::zed_wrapper::start_remote_streamResponse> should match 
// service_traits::DataType< ::zed_wrapper::start_remote_stream > 
template<>
struct DataType< ::zed_wrapper::start_remote_streamResponse>
{
  static const char* value()
  {
    return DataType< ::zed_wrapper::start_remote_stream >::value();
  }
  static const char* value(const ::zed_wrapper::start_remote_streamResponse&)
  {
    return value();
  }
};

} // namespace service_traits
} // namespace ros

#endif // ZED_WRAPPER_MESSAGE_START_REMOTE_STREAM_H