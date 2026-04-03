using Revise
using ENEEGMA
using Plots

settings = create_default_settings()

# change known WC Model to Unknown 
print_settings_summary(settings; section="sampling_settings")
