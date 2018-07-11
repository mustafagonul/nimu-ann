#pragma once

#include <array>
#include <type_traits>


namespace nimu {


template <typename Type, size_t Input, size_t Layer, size_t Hidden, size_t Output>
class ann_static {
private:
  using value_type = Type;
  using size_type = size_t;
  using input_size = Input;
  using layer_size = Layer;
  using hidden_size = Hidden;
  using output_size = Output;

  static_assert(std::is_floating_point<value_type>, "Value type must be floating point");

public:

private:
  using input_type = std::array<value_type, input_size>;
  using layer_type = std::array<value_type, hidden_type>;
  using layers_type = std::array<layer_type, layer_size>;
  using output_type = std::array<value_type, output_size>;

  input_type  input;
  layers_type layers;
  output_type output;
};


}

