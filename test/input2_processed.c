#include <stdint.h>

#define INT_CONST (42)
#define FLOAT_CONST (3.14159f)
#define CHAR_CONST ('A')
#define DERIVED_CONST (INT_CONST * 2)


#define ADD(x,y) ((x) + (y))
#define SQUARE(x) ((x) * (x))
#define PTR_DEREF(p) (*(p))

int main() {

    int i = INT_CONST;
    float f = FLOAT_CONST;
    char c = CHAR_CONST;
    int derived = DERIVED_CONST;
    int arr[] = {INT_CONST, DERIVED_CONST};
    
    int sum = ADD((1), (2));
    float fsum = ADD((1.0f), (2.0f));
    double dsum = ADD((1.0), (2.0));
    
    int sq = SQUARE((3));
    float fsq = SQUARE((3.14f));
    

    int val = 123;
    int* ptr = &val;
    int deref = PTR_DEREF((ptr));
    

    int nums[5] = {1, 2, 3, 4, 5};
    int idx = INT_CONST % 5;
    nums[idx] = SQUARE((nums[ADD((0), (1))]));
    

    int result = INT_CONST > 0 ? SQUARE((i)) : ADD((i), (2));

    
    return sum + sq + deref + result;
}