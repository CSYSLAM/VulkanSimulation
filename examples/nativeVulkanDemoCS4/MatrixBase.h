#pragma once
namespace CsyVk {

class MatrixBase
{
public:
    MatrixBase(){}
    virtual ~MatrixBase(){}
    virtual unsigned int rows() const=0;
    virtual unsigned int cols() const=0;
protected:
};

}  //end of namespace dyno

