# frozen_string_literal: true

require "rouge"

languages = %w[python bash shell yaml json toml dockerfile cpp c cuda javascript typescript go]
missing = languages.reject { |language| Rouge::Lexer.find(language) }

if missing.empty?
  puts "Rouge lexers available: #{languages.join(', ')}"
else
  warn "Missing Rouge lexers: #{missing.join(', ')}"
  exit 1
end
