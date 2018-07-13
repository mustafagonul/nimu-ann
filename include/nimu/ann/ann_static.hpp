#pragma once

#include <array>
#include <type_traits>


namespace nimu {


template <typename Type, size_t Input, size_t Layer, size_t Hidden, size_t Output>
class ann_static {
public:
  using value_type = Type;
  using reference_type = value_type&;
  using const_reference_type = value_type const&;
  using size_type = size_t;

  static constexpr size_type input_size = Input;
  static constexpr size_type layer_size = Layer;
  static constexpr size_type hidden_size = Hidden;
  static constexpr size_type output_size = Output;

  static_assert(std::is_floating_point<value_type>::value, "Value type must be floating point");

public:
  ann_static() = default;
  ~ann_static() = default;

  ann_static(ann_static const&) = default;
  ann_static(ann_static &&) = delete;

  ann_static const& operator=(ann_static const &) = default;
  ann_static&& operator=(ann_static&&) = delete;


  const_reference_type
  input(size_type) const;

  reference_type
  input(size_type);

  const_reference_type
  output(size_type) const;

  reference_type
  output(size_type);

  const_reference_type
  node(size_type, size_type) const;

  reference_type
  node(size_type, size_type);


private:
  using input_type = std::array<value_type, input_size>;
  using layer_type = std::array<value_type, hidden_size>;
  using layers_type = std::array<layer_type, layer_size>;
  using output_type = std::array<value_type, output_size>;

  input_type  input_array;
  layers_type layer_array;
  output_type output_array;
};


}

